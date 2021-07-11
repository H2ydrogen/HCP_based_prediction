import argparse
import logging
import time

LOG = logging.getLogger('main')
__all__ = []


def create_parser():
    parser = argparse.ArgumentParser(description='DTI analysis')
    #data
    parser.add_argument('--data-path', default='D:\Datasets\HCP_S1200', type=str)
    # training
    parser.add_argument('--epochs', default=120, type=int)

    return parser




