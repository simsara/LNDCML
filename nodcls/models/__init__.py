from importlib import import_module


def get_model(model_name):
    return import_module('%s.%s' % (__package__, model_name))


def get_common_config():
    return {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (16, 32, 24, 128)
    }
