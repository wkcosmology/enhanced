
import numpy as np
import matplotlib.pyplot as plt
from enhanced.plot.scatter import Plot2DScatter

def test_plto2dscatter():
    xs = np.random.random(1000)
    ys = np.random.random(1000) * 10 + 10
    fig = plt.figure(figsize=(10, 10))

    scatter = Plot2DScatter(fig, np.linspace(0, 1, 10), np.linspace(10, 20, 10))
    scatter.add_component(xs, ys, color='k')

    xs = np.random.random(1000)
    ys = np.random.random(1000) * 10 + 10
    scatter.add_component(xs, ys, color='r')
    plt.show()


def test_plot2dpercentile():
    import numpy as np
    import matplotlib.pyplot as plt
    from enhanced.plot.scatter import plot_2d_percentile
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = []
    for i in range(5):
        x_tmp = np.random.random(500) + i
        y_tmp = np.random.exponential(scale=1, size=500) + np.random.random(1) * i
        data.append(np.column_stack((x_tmp, y_tmp)))
    data = np.concatenate(data)

    x_edges = np.linspace(0, 5, 6)
    percens = np.array([25, 50, 75])
    plot_2d_percentile(ax, data.T[0], data.T[1], x_edges, percens)
    plt.show()
