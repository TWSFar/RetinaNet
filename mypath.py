class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'voc':
            return '/home/twsf/work/DSSD/data/VOC2012'
        elif dataset == 'coco':
            return '/home/twsf/data/COCO'
        elif dataset == 'visdrone':
            return '/home/twsf/data/Visdrone'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
