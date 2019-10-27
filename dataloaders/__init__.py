from dataloaders.datasets import coco, visdrone
from torch.utils.data import DataLoader


def make_data_loader(opt, train=True):

    if opt.dataset in ['coco', 'COCO', 'Coco']:
        batch_size = opt.batch_size

        if train:
            set_name = 'train2017'
        else:
            set_name = 'val2017'
            batch_size = 1

        dataset = coco.CocoDataset(opt, set_name=set_name, train=train)
        sampler = coco.AspectRatioBasedSampler(
            dataset,
            batch_size=batch_size,
            drop_last=False)
        dataloader = DataLoader(dataset,
                                num_workers=opt.workers,
                                collate_fn=dataset.collater,
                                batch_sampler=sampler)

        return dataset, dataloader

    elif opt.dataset in ['visdrone', 'VisDrone', 'Visdrone']:
        batch_size = opt.batch_size

        if train:
            set_name = 'train'
        else:
            set_name = 'val'
            batch_size = 1

        dataset = visdrone.VisdroneDataset(opt, set_name=set_name, train=train)
        sampler = visdrone.AspectRatioBasedSampler(
            dataset,
            batch_size=batch_size,
            drop_last=False)
        dataloader = DataLoader(dataset,
                                num_workers=opt.workers,
                                collate_fn=dataset.collater,
                                batch_sampler=sampler)

        return dataset, dataloader

    else:
        raise NotImplementedError
