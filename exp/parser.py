#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: CWN project authors 
@author: On Oversquashing project authors 
"""


from distutils.util import strtobool
import argparse


def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')


def get_parser():
    parser = argparse.ArgumentParser()

    # Optimisation params
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'],
                                                     default='acc')
    parser.add_argument('--min_acc', type=float, default=0.5)

    # Model configuration
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=5)
    parser.add_argument('--norm', type=str, default="BatchNorm")
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_act', dest='use_act', type=str2bool,
                        default=True)
    parser.add_argument('--activ', dest='activ', type=str, choices=['tanh',
                                                                      'relu',
                                                                      'elu',
                                                                      'selu',
                                                                      'lrelu',
                                                                      'gelu'],
                                                                default='elu')

    parser.add_argument('--reduce',type=str, choices=["sum", "mul",
                                                      "mean", "min" , "max"],
                                                                default='sum')
    parser.add_argument('--model',  type=str,
                        choices=['gcn','gat','sage','gin'], default='gcn')
    parser.add_argument('--mpnn_layers', type=int, default=2)
    # Experiment parameters
    parser.add_argument('--dataset', type=str,  choices=['TREE','LOLLIPOP','RING'], default='RING')
    parser.add_argument('--seed', type=int, default=808)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--entity', type=str, default="none")

    # Synthetic experiments settings
    parser.add_argument('--add_crosses', type=str2bool, default=False)
    parser.add_argument('--synth_train_size', type=int, default=5000)
    parser.add_argument('--synth_test_size', type=int, default=500)
    parser.add_argument('--synthetic_size', type=int, default=10)
    parser.add_argument('--generate_tree', type=str2bool, default=False)
    parser.add_argument('--arity', type=int, default=2)
    parser.add_argument('--num_class', type=int, default=5)

    return parser
