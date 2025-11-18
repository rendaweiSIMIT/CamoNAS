from dataloaders.datasets import camo, cod10k, chameleon, nc4k
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data.distributed


def make_data_loader(args, **kwargs):
    if args.dataset != 'camonas_cod':
        raise NotImplementedError

    if args.dist:
        print("=> Using Distribued Sampler")

        camo_train = camo.CAMODataset(args, split='train')
        cod10k_train = cod10k.COD10KDataset(args, split='train')
        num_class = camo_train.NUM_CLASSES

        full_train = ConcatDataset([camo_train, cod10k_train])

        train_len = len(full_train)
        split_len = train_len // 2
        lengths = [split_len, train_len - split_len]
        train_set1, train_set2 = torch.utils.data.random_split(full_train, lengths)

        sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1)
        sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2)

        train_loader1 = DataLoader(
            train_set1,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler1,
            **kwargs
        )
        train_loader2 = DataLoader(
            train_set2,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler2,
            **kwargs
        )

        chameleon_test = chameleon.CHAMELEONDataset(args, split='test')
        camo_test = camo.CAMODataset(args, split='test')
        cod10k_test = cod10k.COD10KDataset(args, split='test')
        nc4k_test = nc4k.NC4KDataset(args, split='test')

        val_set = ConcatDataset([chameleon_test, camo_test, cod10k_test, nc4k_test])
        test_set = ConcatDataset([chameleon_test, camo_test, cod10k_test, nc4k_test])

        sampler_val = torch.utils.data.distributed.DistributedSampler(val_set)
        sampler_test = torch.utils.data.distributed.DistributedSampler(test_set)

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler_val,
            **kwargs
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler_test,
            **kwargs
        )

        return train_loader1, train_loader2, val_loader, test_loader, num_class

    else:
        camo_train = camo.CAMODataset(args, split='train')
        cod10k_train = cod10k.COD10KDataset(args, split='train')
        num_class = camo_train.NUM_CLASSES

        full_train = ConcatDataset([camo_train, cod10k_train])

        train_len = len(full_train)
        split_len = train_len // 2
        lengths = [split_len, train_len - split_len]
        train_set1, train_set2 = torch.utils.data.random_split(full_train, lengths)

        train_loader1 = DataLoader(
            train_set1,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        train_loader2 = DataLoader(
            train_set2,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

        chameleon_test = chameleon.CHAMELEONDataset(args, split='test')
        camo_test = camo.CAMODataset(args, split='test')
        cod10k_test = cod10k.COD10KDataset(args, split='test')
        nc4k_test = nc4k.NC4KDataset(args, split='test')

        val_set = ConcatDataset([chameleon_test, camo_test, cod10k_test, nc4k_test])
        test_set = ConcatDataset([chameleon_test, camo_test, cod10k_test, nc4k_test])

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )

        return train_loader1, train_loader2, val_loader, test_loader, num_class
