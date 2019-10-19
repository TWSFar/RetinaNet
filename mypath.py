class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'voc':
            return '/home/twsf/work/DSSD/data/VOC2012'
        elif dataset == 'coco':
            return '/home/twsf/data/COCO'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

