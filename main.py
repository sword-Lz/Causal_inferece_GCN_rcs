import argparse
import logging
import os
import random
import numpy as np
import torch

# import torch.backends.cudnn as cudnn

from dataset.load_data import rcs_dataset
from model.NGCF import NGCF
from trainer import trainer
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/',
                    help='root dir for data')
parser.add_argument('--root_path', type=str,
                    default='./data/',
                    help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='amazon-book', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='NGCF', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--epoch', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1024, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
args = parser.parse_args()

if __name__ == "__main__":
    # if not args.deterministic:
    #    cudnn.benchmark = True
    #    cudnn.deterministic = False
    # else:
    #    cudnn.benchmark = False
    #    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    args.exp = dataset_name
    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer(args, snapshot_path)