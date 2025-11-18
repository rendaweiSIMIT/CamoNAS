import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from config_utils.search_args import obtain_search_args
from utils.copy_state_dict import copy_state_dict
import apex

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))

torch.backends.cudnn.benchmark = True


class LDWT(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.a0 = nn.Parameter(torch.randn(kernel_size))
        self.a1 = nn.Parameter(torch.randn(kernel_size))

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.kernel_size
        pad = k // 2
        a0 = self.a0
        a1 = self.a1
        f_ll = torch.ger(a0, a0)
        f_lh = torch.ger(a0, a1)
        f_hl = torch.ger(a1, a0)
        f_hh = torch.ger(a1, a1)
        kernels = torch.stack([f_ll, f_lh, f_hl, f_hh], dim=0)
        kernels = kernels.unsqueeze(1)
        kernels = kernels.repeat(c, 1, 1, 1)
        x_ = x.view(b * c, 1, h, w)
        out = F.conv2d(x_, kernels, stride=2, padding=pad)
        _, _, h2, w2 = out.shape
        out = out.view(b, c * 4, h2, w2)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=2):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SobelFilter(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        kernel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32,
        )
        kernel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32,
        )
        weight = torch.stack([kernel_x, kernel_y], dim=0)
        weight = weight.unsqueeze(1)
        self.register_buffer('weight', weight)
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        w_ = self.weight.repeat(c, 1, 1, 1)
        x_ = x.view(b * c, 1, h, w)
        out = F.conv2d(x_, w_, stride=self.stride, padding=1)
        _, _, h2, w2 = out.shape
        out = out.view(b, c, 2, h2, w2).mean(dim=2)
        return out


class HaarSplit(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        h = torch.tensor([1.0, 1.0], dtype=torch.float32) / 2.0
        g = torch.tensor([1.0, -1.0], dtype=torch.float32) / 2.0
        f_ll = torch.ger(h, h)
        f_lh = torch.ger(h, g)
        f_hl = torch.ger(g, h)
        f_hh = torch.ger(g, g)
        kernels = torch.stack([f_ll, f_lh, f_hl, f_hh], dim=0)
        kernels = kernels.unsqueeze(1)
        self.register_buffer('kernels', kernels)
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.kernels
        k_ = k.repeat(c, 1, 1, 1)
        x_ = x.view(b * c, 1, h, w)
        out = F.conv2d(x_, k_, stride=self.stride, padding=1)
        _, _, h2, w2 = out.shape
        out = out.view(b, c, 4, h2, w2).mean(dim=2)
        return out


class GaussianBlur(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        kernel = torch.tensor(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]],
            dtype=torch.float32,
        )
        kernel = kernel / kernel.sum()
        weight = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer('weight', weight)
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        w_ = self.weight.repeat(c, 1, 1, 1)
        x_ = x.view(b * c, 1, h, w)
        out = F.conv2d(x_, w_, stride=self.stride, padding=1)
        _, _, h2, w2 = out.shape
        out = out.view(b, c, h2, w2)
        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


def build_op(name, channels, stride):
    if name == 'sep_3x3':
        return DepthwiseSeparableConv(channels, 3, stride=stride, dilation=1)
    if name == 'sep_5x5':
        return DepthwiseSeparableConv(channels, 5, stride=stride, dilation=1)
    if name == 'dil_3x3':
        return DilatedConv(channels, 3, stride=stride, dilation=2)
    if name == 'dil_5x5':
        return DilatedConv(channels, 5, stride=stride, dilation=2)
    if name == 'avg_pool_3x3':
        return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    if name == 'max_pool_3x3':
        return nn.MaxPool2d(3, stride=stride, padding=1)
    if name == 'skip_connect':
        if stride == 1:
            return Identity()
        else:
            return nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
    if name == 'none':
        return Zero(stride=stride)
    if name == 'sobel':
        return SobelFilter(channels, stride=stride)
    if name == 'haar':
        return HaarSplit(channels, stride=stride)
    if name == 'gaussian':
        return GaussianBlur(channels, stride=stride)
    raise ValueError('Unknown op {}'.format(name))


class MixedOp(nn.Module):
    def __init__(self, channels, op_names, stride):
        super().__init__()
        self.ops = nn.ModuleList()
        for name in op_names:
            op = build_op(name, channels, stride)
            self.ops.append(op)

    def forward(self, x, weights):
        out = 0.0
        for w, op in zip(weights, self.ops):
            out = out + w * op(x)
        return out


class CamoNASCell(nn.Module):
    def __init__(self, channels, op_names, steps):
        super().__init__()
        self.steps = steps
        self.ops = nn.ModuleList()
        for _ in range(steps):
            self.ops.append(MixedOp(channels, op_names, stride=1))

    def forward(self, x, weights):
        out = x
        for i in range(self.steps):
            w = weights[i]
            out = self.ops[i](out, w)
        return out


class SoftVQFusionHead(nn.Module):
    def __init__(self, in_channels, embed_dim, codebook_size, num_classes, temperature=0.1):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.codebook = nn.Parameter(torch.randn(embed_dim, codebook_size))
        self.temperature = temperature
        self.proj_out = nn.Conv2d(embed_dim, in_channels, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, rgb_feats, freq_feats):
        feats = rgb_feats + freq_feats
        base = feats[0]
        b, _, h, w = base.shape
        aligned = []
        for f in feats:
            if f.shape[2:] != (h, w):
                f = F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
            aligned.append(f)
        x = torch.cat(aligned, dim=1)
        z = self.proj_in(x)
        b, d, h, w = z.shape
        n = h * w
        z_flat = z.view(b, d, n)
        z_norm = F.normalize(z_flat, dim=1)
        c_norm = F.normalize(self.codebook, dim=0)
        scores = torch.einsum('bdn,dk->bkn', z_norm, c_norm)
        scores = scores / self.temperature
        membership = F.softmax(scores, dim=1)
        z_hat = torch.einsum('dk,bkn->bdn', self.codebook, membership)
        z_hat = z_hat.view(b, d, h, w)
        y = self.proj_out(z_hat) + x
        logits = self.classifier(y)
        return logits


class CamoNAS(nn.Module):
    def __init__(self, num_classes, num_layers, criterion, filter_multiplier, block_multiplier, step):
        super().__init__()
        base_channels = filter_multiplier
        self.num_layers = num_layers
        self.step = step
        self.op_names = [
            'sep_3x3',
            'sep_5x5',
            'dil_3x3',
            'dil_5x5',
            'avg_pool_3x3',
            'max_pool_3x3',
            'skip_connect',
            'none',
            'sobel',
            'haar',
            'gaussian',
        ]
        self.ldwt = LDWT(in_channels=3, kernel_size=3)
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        freq_in_channels = 3 * 4
        self.freq_stem = nn.Sequential(
            nn.Conv2d(freq_in_channels, base_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.rgb_cells = nn.ModuleList()
        self.freq_cells = nn.ModuleList()
        self.alpha = nn.ParameterList()
        for _ in range(num_layers):
            cell_rgb = CamoNASCell(base_channels, self.op_names, step)
            cell_freq = CamoNASCell(base_channels, self.op_names, step)
            self.rgb_cells.append(cell_rgb)
            self.freq_cells.append(cell_freq)
            num_edges = step
            param = nn.Parameter(1e-3 * torch.randn(num_edges, len(self.op_names)))
            self.alpha.append(param)
        fusion_in_channels = base_channels * 4
        self.fusion_head = SoftVQFusionHead(
            fusion_in_channels,
            embed_dim=filter_multiplier * block_multiplier,
            codebook_size=32,
            num_classes=num_classes,
        )

    def forward(self, x):
        h_in, w_in = x.shape[2], x.shape[3]
        freq = self.ldwt(x)
        f_rgb = self.rgb_stem(x)
        freq_down = F.interpolate(freq, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
        f_freq = self.freq_stem(freq_down)
        rgb_feats = []
        freq_feats = []
        for i in range(self.num_layers):
            weights = F.softmax(self.alpha[i], dim=-1)
            f_rgb = self.rgb_cells[i](f_rgb, weights)
            f_freq = self.freq_cells[i](f_freq, weights)
            if i in [self.num_layers // 4 - 1, self.num_layers // 2 - 1]:
                rgb_feats.append(f_rgb)
                freq_feats.append(f_freq)
        if len(rgb_feats) < 2:
            rgb_feats.append(f_rgb)
            freq_feats.append(f_freq)
        logits = self.fusion_head(rgb_feats, freq_feats)
        logits = F.interpolate(logits, size=(h_in, w_in), mode='bilinear', align_corners=False)
        return logits

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'alpha' not in name:
                yield param

    def arch_parameters(self):
        for p in self.alpha:
            yield p


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader, self.nclass = make_data_loader(
            args, **kwargs
        )
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(
                Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy'
            )
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                raise NotImplementedError
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        model = CamoNAS(
            self.nclass,
            12,
            self.criterion,
            self.args.filter_multiplier,
            self.args.block_multiplier,
            self.args.step,
        )
        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        self.model, self.optimizer = model, optimizer
        self.architect_optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_lr,
            betas=(0.9, 0.999),
            weight_decay=args.arch_weight_decay,
        )
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_loaderA), min_lr=args.min_lr
        )
        if args.cuda:
            self.model = self.model.cuda()
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        if module.weight is None:
                            module.weight = torch.nn.Parameter(
                                torch.ones(
                                    module.running_var.shape,
                                    dtype=module.running_var.dtype,
                                    device=module.running_var.device,
                                ),
                                requires_grad=False,
                            )
                        if module.bias is None:
                            module.bias = torch.nn.Parameter(
                                torch.zeros(
                                    module.running_var.shape,
                                    dtype=module.running_var.dtype,
                                    device=module.running_var.device,
                                ),
                                requires_grad=False,
                            )
            self.model, [self.optimizer, self.architect_optimizer] = amp.initialize(
                self.model,
                [self.optimizer, self.architect_optimizer],
                opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32,
                loss_scale="dynamic",
            )
            print('cuda finished')
        if args.cuda and len(self.args.gpu_ids) > 1:
            if self.opt_level == 'O2' or self.opt_level == 'O3':
                print('currently cannot run with nn.DataParallel and optimization level', self.opt_level)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            print('training on multiple-GPUs')
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                copy_state_dict(self.model.state_dict(), new_state_dict)
            else:
                if torch.cuda.device_count() > 1 or args.load_parallel:
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
                else:
                    copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])
            if not args.ft:
                copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loaderA)
        num_img_tr = len(self.train_loaderA)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            if epoch >= self.args.alpha_epoch:
                try:
                    search = next(self.search_iter)
                except AttributeError:
                    self.search_iter = iter(self.train_loaderB)
                    search = next(self.search_iter)
                except StopIteration:
                    self.search_iter = iter(self.train_loaderB)
                    search = next(self.search_iter)
                image_search, target_search = search['image'], search['label']
                if self.args.cuda:
                    image_search, target_search = image_search.cuda(), target_search.cuda()
                self.architect_optimizer.zero_grad()
                output_search = self.model(image_search)
                arch_loss = self.criterion(output_search, target_search)
                if self.use_amp:
                    with amp.scale_loss(arch_loss, self.architect_optimizer) as arch_scaled_loss:
                        arch_scaled_loss.backward()
                else:
                    arch_loss.backward()
                self.architect_optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            if i % (num_img_tr // 10 + 1) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(
                    self.writer, self.args.dataset, image, target, output, global_step
                )
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        if self.args.no_val:
            is_best = False
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                },
                is_best,
            )

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)
        acc = self.evaluator.Pixel_Accuracy()
        acc_class = self.evaluator.Pixel_Accuracy_Class()
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', miou, epoch)
        self.writer.add_scalar('val/Acc', acc, epoch)
        self.writer.add_scalar('val/Acc_class', acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', fwiou, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(acc, acc_class, miou, fwiou))
        print('Loss: %.3f' % test_loss)
        new_pred = miou
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': state_dict,
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                },
                is_best,
            )


def main():
    args = obtain_search_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 40,
            'pascal': 50,
            'kd': 10,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    if args.checkname is None:
        args.checkname = 'camonas'
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    trainer.writer.close()


if __name__ == "__main__":
    main()
