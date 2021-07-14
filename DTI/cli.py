import argparse
import logging
import time

LOG = logging.getLogger('main')
__all__ = []


def create_parser():
    parser = argparse.ArgumentParser(description='DTI analysis')
    # data
    parser.add_argument('--data-path', default='D:\\Datasets\\HCP_S1200', type=str)
    parser.add_argument('--INPUT-FEATURES', default=['FA1-mean', 'FA2-mean', 'Trace1-mean', 'Trace1-mean'][0], type=str)
    parser.add_argument('--FEATURES-TYPE', default=['right-hemisphere', 'left-hemisphere',
                                                    'anatomical-hemisphere', 'commissural'][0], type=str)
    # Network
    parser.add_argument('--MODEL', default='1D-CNN', type=str)
    # training
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--display-batch', default=50, type=int)
    parser.add_argument('--LR', default=0.1, type=float)
    # record
    parser.add_argument('--RECORD-PATH', default='.\\LOG\\{}.txt'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime())), type=str)

    return parser