import inspect
from matplotlib import axes
from collections import namedtuple

_Arg = namedtuple('Arg', ['value', 'data_arg', 'parameter'])


class Brush(object):
    def __init__(self, method=None, args={}, data_args={}):
        self.__method = getattr(axes.Axes, method)
        self.__all_args = {}

        # analyse method parameters
        parameters = inspect.signature(self.__method).parameters
        i = -1
        for name, param in parameters.items():
            # drop self, *args, **kwargs
            i += 1
            if i == 0 or \
                param.kind is inspect._VAR_POSITIONAL or \
                param.kind is inspect._VAR_KEYWORD : continue

            # found name in args
            data_arg = False
            value = None
            if name in args: value = args[name]
            elif name in data_args:
                value = data_args[name]
                data_arg = True

            if value is None:
                if param.default is inspect._empty: raise Exception('the argument {} should be supplied in method'.format(name, method))
                else: continue

            self.__all_args[name] = _Arg(value, data_arg, param)



    @property
    def method(self): return self.__method

    def parse_kwargs(self, datas):
        kwargs = {}
        for name, arg in self.__all_args.items():
            required = arg.parameter.default == inspect._empty
            v = arg.value
            if arg.data_arg: v = datas[arg.value] if arg.value in datas else None
            if v is None and required: return False, None
            kwargs[name] = v
        return True, kwargs