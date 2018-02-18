import matplotlib.pyplot as plt

class Plot(object):
    def __init__(self, axes=None, brushes=[]):
        self.__axes = axes or plt.subplot()
        self.__brushes = brushes
        self.__datas = None
        self.__event_dict = {}

    def erase(self): self.__axes.cla()

    def paint(self, datas, erase_before=False):
        if erase_before: self.erase()
        self.__datas = datas
        if self.__datas is None: return
        for b in self.__brushes:
            if not b.is_valid(self.__datas): continue
            f = self.__axes[b.method]
            f(data=self.__datas, **b.kwargs)

    def repaint(self, datas): self.paint(datas, erase_before=True)

    def show(self, datas, erase_before=False):
        self.paint(datas, erase_before)
        plt.show()

    def connect_event(self, name, functor):
        def __listener(event):
            if event.inaxes != self.__axes: return
            functor()
        return self.__axes.get_figure().canvas.mpl_connect(name, __listener)

    def disconnect_event(self, id): self.__axes.get_figure().canvas.mpl_disconnect(id)

