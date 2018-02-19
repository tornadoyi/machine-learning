import inspect
from types import FunctionType, MethodType
from matplotlib import axes

class Brush(object):
    def __init__(self, method=None, **kwargs):
        self.__method = getattr(axes.Axes, method)
        self.__kwargs = kwargs
        self.__required_params = []
        self.__required_data_key = []

        # analyse method parameters
        sig = inspect.signature(self.__method)
        for name, param in sig.parameters.items():
            # find required parameters
            if param.name == 'self' or param.name == 'kwargs' or param.default is not inspect._empty: continue
            self.__required_params.append(name)

            # check all required paramters exist in kwargs
            if name not in self.__kwargs: raise Exception('{} is a required parameter in function {}'.format(name, method))

            self.__required_data_key.append(self.__kwargs[name])


    @property
    def method(self): return self.__method

    @property
    def kwargs(self): return self.__kwargs

    def is_valid(self, datas):
        for key in self.__required_data_key:
            if key not in datas: return False
        return True
