

class Brush(object):
    def __init__(self, method=None, **kwargs):
        self.__method = method
        self.__kwargs = kwargs


    @property
    def method(self): return self.__method

    @property
    def kwargs(self): return self.__kwargs

    def is_valid(self, datas):
        for arg_name, data_key in datas.items():
            if data_key not in datas: return False
        return True