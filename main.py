from parameters import parameters
from train import train
# from predict import predict
from datapreparaion import datapreparation
# from networks.mycrnn import *
from networks.mylstm import *
import argparse
import os

# parse arguments
parser = argparse.ArgumentParser(description='CTC')
parser.add_argument('-run', default=0, type=int)
args = parser.parse_args()

# MAIN CODE
# load parameters
params = parameters()
params.modify('parameters/DB3.txt') # DIY parameters in parameters/*.txt and load here
# params.print_params()

# data preparation
data_folds, target_folds = datapreparation(params)

params.print_params()
# build network
network = mylstm(params)

train(params, network, data_folds, target_folds)

# if args.run == 0:
#     from train import train
#     train(params, network, data_folds, target_folds)
# elif args.run == 1:
#     predict(params, network)
# elif args.run == 2:
#     pass
# else:
#     raise ValueError("Undefined -run %s." % str(args.run))

