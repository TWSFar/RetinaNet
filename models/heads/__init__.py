from .retina_head import RetinaHead
from .fcos_head import FCOSHead


def build_head(args):
    obj_type = args.pop('type')

    if obj_type == "RetinaHead":
        return RetinaHead(**args)

    if obj_type == "FCOSHead":
        return FCOSHead(**args)

    else:
        raise NotImplementedError
