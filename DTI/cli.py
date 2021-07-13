import argparse
import logging
import time

LOG = logging.getLogger('main')
__all__ = []


def create_parser():
    parser = argparse.ArgumentParser(description='DTI analysis')
    # data
    parser.add_argument('--data-path', default='D:\\Datasets\\HCP_S1200', type=str)
    # training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--display-batch', default=10, type=int)

    return parser
