import numpy as np
from .plot import Plot
from .brush import Brush

def create_cluster_plot(button_press_event=None):
    def data_processor(datas):
        if datas is None: return datas
        num_colors = datas['num_clusters'] if 'num_clusters' in datas else 1
        colors = np.linspace(0x0000ff, 0xff0000, num_colors, dtype=np.int)
        if 'point_color' not in datas:
            labels = datas.get('labels', None)
            if labels is None:
                datas['point_color'] = '#{0:06x}'.format(colors[0])
            else:
                datas['point_color'] = ['#{0:06x}'.format(colors[l]) for l in labels]

        if 'center_color' not in datas:
            datas['center_color'] = ['#{0:06x}'.format(c) for c in colors]

        return datas

    plot = Plot(brushes=[
            # points brush
            Brush(method='scatter', x='point_x', y='point_y', c='point_color', linestyle='None', marker='o'),

            # centers brush
            Brush(method='scatter', x='center_x', y='center_y', c='center_color', linestyle='None', marker='*'),
        ],
        data_processor=data_processor)

    if button_press_event is not None:
        plot.connect_event('button_press_event', button_press_event)

    return plot




