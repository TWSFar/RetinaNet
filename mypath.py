import os
user_dir = os.path.expanduser('~')


class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'voc':
            return user_dir + '/work/DSSD/data/VOC2012'
        elif dataset == 'coco':
            return user_dir + '/data/COCO'
        elif dataset == 'visdrone':
            return user_dir + '/data/Visdrone'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
