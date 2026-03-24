# Source Generated with Decompyle++
# File: losses.cpython-310.pyc (Python 3.10)


def weighted_mse(x_hat, x, threshold=0.1, high_weight=20.0):
    mask = (x > threshold).float()
    weights = 1.0 + mask * high_weight
    loss = (x_hat - x) ** 2 * weights
    return loss.mean()

