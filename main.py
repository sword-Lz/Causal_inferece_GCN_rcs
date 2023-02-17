import os
import random
import numpy as np
import torch

# import torch.backends.cudnn as cudnn

from train.trainer_NGCF import *
import warnings
from utility.parser import parse_args_NGCF

args = parse_args_NGCF()
warnings.filterwarnings('ignore')


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
    snapshot_path = "../model/{}/{}".format(args.exp, 'NGCF')
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 100000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer_cause_NGCF(args, snapshot_path)