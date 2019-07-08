def test_empiricalinterp2d():
    from enhanced.stats.binned_stats import EmpiricalInterp2d
    import numpy as np

    def func(xs, ys):
        return (xs - 0.5)**2 + (ys - 0.5)**2

    xs_in = np.random.random(10000)
    ys_in = np.random.random(10000)
    vals_in = func(xs_in, ys_in)

    x_edges = np.linspace(0, 1, 10)
    y_edges = np.linspace(0, 1, 10)
    test = EmpiricalInterp2d(xs_in, ys_in, vals_in, x_edges, y_edges)

    test_xs = np.random.random(100)
    test_ys = np.random.random(100)
    predict = test.interp(test_xs, test_ys)
    true = func(test_xs, test_ys)
    assert np.all((predict - true) < 0.08)
