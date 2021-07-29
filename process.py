import pandas as pd

def create_bins(lower_bound, width, quantity):
    """ create_bins returns an equal-width (distance) partitioning.
        It returns an ascending list of tuples, representing the intervals.
        A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0
        and i < quantity, satisfies the following conditions:
            (1) bins[i][0] + width == bins[i][1]
            (2) bins[i-1][0] + width == bins[i][0] and
                bins[i-1][1] + width == bins[i][1]
    """

    bins = []
    for low in range(lower_bound,
                     lower_bound + quantity * width + 1, width):
        bins.append((low, low + width))
    return bins


def classify_sale_price(bins, x):
    for idx in range(len(bins)):

        bMn = bins[idx][0]
        bMx = bins[idx][1]
        if bMn < x < bMx:
            cls = idx
            return int(cls)


def get_adjRsq(X: pd.DataFrame, y: pd.Series, rSq):
    p = len(X.columns)
    n = len(y)
    adjRsq = 1 - (1 - rSq) * (n - 1) / (n - p - 1)
    return adjRsq


def normalize(x, mean, stdev):
    z = (x - mean) / stdev
    return z

# def get_sale_class_percent(cls: pd.Series):
#     tmp = {}
#     for c in cls.unique():
#         count = cls[cls['priceClass']]
