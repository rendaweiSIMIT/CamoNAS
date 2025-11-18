import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CHAMELEONDataset(Dataset):
    NUM_CLASSES = 1

    def __init__(self, args, split='test'):
        self.root = args.chameleon_root
        self.split = split
        self.img_dir = os.path.join(self.root, "Imgs")
        self.mask_dir = os.path.join(self.root, "GT")
        self.imgs = sorted(os.listdir(self.img_dir))

        self.transform_img = transforms.ToTensor()
        self.transform_mask = transforms.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        name = self.imgs[index]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name.replace('.jpg', '.png'))

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        return {"image": img, "label": mask}
