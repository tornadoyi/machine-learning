import matplotlib.pyplot as plt

class Plot(object):
    def __init__(self, axes=None, brushes=[], data_processor=None):
        self.__axes = axes or plt.subplot()
        self.__brushes = brushes
        self.__data_processor = data_processor

    def add_brush(self, brush):
        self.__brushes.append(brush)
        return len(self.__brushes) - 1

    def remove_brush(self, index):
        del self.__brushes[index]

    def erase(self): self.__axes.cla()

    def paint(self, datas):
        if self.__data_processor is not None: datas = self.__data_processor(datas)
        if datas is None: return
        for b in self.__brushes:
            valid, kwargs = b.parse_kwargs(datas)
            if not valid: continue
            b.method(self.__axes, **kwargs)

    def show(self, datas=None):
        self.paint(datas)
        plt.show()

    def connect_event(self, name, functor):
        def __listener(event):
            if event.inaxes != self.__axes: return
            functor(event)
        return self.__axes.get_figure().canvas.mpl_connect(name, __listener)

    def disconnect_event(self, id): self.__axes.get_figure().canvas.mpl_disconnect(id)

    def set_data_processor(self, processor): self.__data_processor = processor

    def set_title(self, title): self.__axes.set_title(title)
