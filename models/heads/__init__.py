from .retina_head import RetinaHead


def build_head(args):
    obj_type = args.pop('type')

    if obj_type == "RetinaHead":
        return RetinaHead(**args)

    else:
        raise NotImplementedError
