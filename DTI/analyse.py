import numpy as np
import logging

LOG = logging.getLogger('analyse')


def analyse_3class(target_list, pred_list):
    P00 = ((pred_list == 0) & (target_list == 0)).sum()
    P11 = ((pred_list == 1) & (target_list == 1)).sum()
    P22 = ((pred_list == 2) & (target_list == 2)).sum()
    P01 = ((pred_list == 0) & (target_list == 1)).sum()
    P02 = ((pred_list == 0) & (target_list == 2)).sum()
    P12 = ((pred_list == 1) & (target_list == 2)).sum()
    P10 = ((pred_list == 1) & (target_list == 0)).sum()
    P21 = ((pred_list == 2) & (target_list == 1)).sum()
    P20 = ((pred_list == 2) & (target_list == 0)).sum()
    acc = round((P11 + P22 + P00) / pred_list.size, 4)
    p0 = round(P00 / (P00 + P01 + P02), 4)
    p1 = round(P11 / (P10 + P11 + P12), 4)
    p2 = round(P22 / (P20 + P21 + P22), 4)
    r0 = round(P00 / (P00 + P10 + P20), 4)
    r1 = round(P11 / (P01 + P11 + P21), 4)
    r2 = round(P22 / (P02 + P12 + P22), 4)

    # LOG
    LOG.info('acc:{}, precison:{}/{}/{}, recall:{}/{}/{}'
             .format(acc, r0, r1, r2, p0, p1, p2))

    return {
        'ACC': acc,
        'precision': [p0, p1],
        'recall': [r0, r1]
    }
