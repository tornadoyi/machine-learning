import numpy as np
from .plot import Plot
from .brush import Brush

def create_cluster_plot(button_press_event=None):
    def data_processor(datas):
        if datas is None: return datas

        # calculate total colors
        num_colors = datas['num_clusters'] if 'num_clusters' in datas else 1
        colors = np.linspace(0x0000ff, 0xff0000, num_colors, dtype=np.int)

        # set color of point
        if 'point_color' not in datas:
            labels = datas.get('labels', None)
            alphas = datas.get('point_alpha', None)
            if alphas is not None: alphas = (alphas * 255).astype(np.int)
            rgb = np.array([colors[0]]) if labels is None else np.array([colors[l] for l in labels])
            a = np.array([0xff]) if alphas is None else alphas
            point_color = (rgb << 8) + a
            datas['point_color'] = ['#{0:08x}'.format(c) for c in point_color]

        # set color of center
        if 'center_color' not in datas:
            datas['center_color'] = ['#{0:06x}'.format(c) for c in colors]

        return datas

    plot = Plot(brushes=[
            # points brush
            Brush(method='scatter',
                  data_args={'x':'point_x', 'y':'point_y', 'c': 'point_color'},
                  args={'linestyle':'None', 'marker':'o'}),

            # centers brush
            Brush(method='scatter',
                  data_args={'x':'center_x', 'y':'center_y', 'c':'center_color'},
                  args={'linestyle':'None', 'marker':'*'}),
        ],
        data_processor=data_processor)

    if button_press_event is not None:
        plot.connect_event('button_press_event', button_press_event)

    return plot




