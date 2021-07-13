import numpy as np


def analyse_3class(targetlist, predlist):
    P00 = ((predlist == 0) & (targetlist == 0)).sum()
    P11 = ((predlist == 1) & (targetlist == 1)).sum()
    P22 = ((predlist == 2) & (targetlist == 2)).sum()
    P01 = ((predlist == 0) & (targetlist == 1)).sum()
    P02 = ((predlist == 0) & (targetlist == 2)).sum()
    P12 = ((predlist == 1) & (targetlist == 2)).sum()
    P10 = ((predlist == 1) & (targetlist == 0)).sum()
    P21 = ((predlist == 2) & (targetlist == 1)).sum()
    P20 = ((predlist == 2) & (targetlist == 0)).sum()
    acc = (P11 + P22 + P00) / predlist.size
    p0 = P00 / (P00 + P01 + P02)
    p1 = P11 / (P10 + P11 + P12)
    p2 = P22 / (P20 + P21 + P22)
    r0 = P00 / (P00 + P10 + P20)
    r1 = P11 / (P01 + P11 + P21)
    r2 = P22 / (P02 + P12 + P22)

    return {
        'ACC': acc,
        'precision': [p0, p1, p2],
        'recall': [r0, r1, r2]
    }
