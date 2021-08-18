import argparse
import logging
import time

LOG = logging.getLogger('main')
__all__ = []


def create_parser():
    parser = argparse.ArgumentParser(description='DTI analysis')
    # data
    parser.add_argument('--data-path', default='D:\\Datasets\\HCP_S1200', type=str)
    parser.add_argument('--INPUT-FEATURES', default=['FA1-min', 'FA1-max', 'FA1-mean', 'Num_Fibers', 'FA2-mean',
                                                     'Trace1-mean', 'Trace2-mean', ][2:5], type=list)
    parser.add_argument('--HEMISPHERES', default=['right-hemisphere', 'left-hemisphere', 'anatomical',
                                                  'commissural', ][0:3], type=list)
    parser.add_argument('--OUTPUT-FEATURES', default=['sex', 'age', 'race', 'hand', 'BMI'][0], type=str)
    parser.add_argument('--NUM_CLASSES', default=2, type=int)
    parser.add_argument('--FOLD-NUM', default=5, type=int)
    # Network
    parser.add_argument('--MODEL', default=['1D-CNN', '2D-CNN', 'Lenet'][0], type=str)
    parser.add_argument('--LOAD_PATH', default='C:\\Users\\admin\\PycharmProjects\\dti\\LOG\\{}.pkl', type=str)
    # training
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--display-batch', default=50, type=int)
    parser.add_argument('--LR', default=0.01, type=float)
    parser.add_argument('--L2', default=0, type=float)
    # record
    parser.add_argument('--RECORD-NAME', default='{}'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime())), type=str)

    return parser