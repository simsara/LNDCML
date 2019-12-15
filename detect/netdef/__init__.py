from importlib import import_module


def get_model(model_name):
    return import_module('%s.%s' % (__package__, model_name))
