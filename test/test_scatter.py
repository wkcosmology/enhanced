
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
