from .datasets import (
    visdrone, visdrone_chip_json, visdrone_chip_xml)
from torch.utils.data import DataLoader


def make_data_loader(opt, train=True):
    if str.lower(opt.dataset) == 'visdrone':
        batch_size = opt.batch_size

        if train:
            set_name = 'train'
        else:
            set_name = 'val'

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

    elif str.lower(opt.dataset) == 'visdrone_chip_json':
        batch_size = opt.batch_size

        if train:
            set_name = 'train'
        else:
            set_name = 'val'

        dataset = visdrone_chip_json.VisdroneDataset(opt, set_name=set_name, train=train)
        sampler = visdrone_chip_json.AspectRatioBasedSampler(
            dataset,
            batch_size=batch_size,
            drop_last=False)
        dataloader = DataLoader(dataset,
                                num_workers=opt.workers,
                                collate_fn=dataset.collater,
                                batch_sampler=sampler)

        return dataset, dataloader

    elif str.lower(opt.dataset) == 'visdrone_chip_xml':
        batch_size = opt.batch_size

        if train:
            set_name = 'train'
        else:
            set_name = 'val'

        dataset = visdrone_chip_xml.VisdroneDataset(opt, set_name=set_name, train=train)
        sampler = visdrone_chip_xml.AspectRatioBasedSampler(
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
