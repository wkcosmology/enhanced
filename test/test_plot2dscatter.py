
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


def test_plot2dscatterbinnedhist():
    import numpy as np
    import matplotlib.pyplot as plt
    from enhanced.plot.scatter import Plot2DScatterWithBinnedHist
    fig = plt.figure(figsize=(10, 10))
    data = []
    for i in range(5):
        x_tmp = np.random.random(1000) + i
        y_tmp = np.random.normal(scale=1, size=1000) + np.random.random(1) * i
        data.append(np.column_stack((x_tmp, y_tmp)))
    data = np.concatenate(data)

    data_2 = []
    for i in range(5):
        x_tmp = np.random.random(1000) + i
        y_tmp = np.random.normal(scale=1, size=1000) + 4 - np.random.random(1) * i
        data_2.append(np.column_stack((x_tmp, y_tmp)))
    data_2 = np.concatenate(data_2)

    test = Plot2DScatterWithBinnedHist(fig, np.linspace(0, 4, 5), np.linspace(0, 4, 5), x_label="x Axis", y_label="y Axis")
    test.add_component(data.T[0], data.T[1], np.linspace(0, 4, 15), np.linspace(0, 4, 15), color='b', label='blue population')
    test.add_component(data_2.T[0], data_2.T[1], np.linspace(0, 4, 15), np.linspace(0, 4, 15), color='r', label='red population')
    plt.show()
