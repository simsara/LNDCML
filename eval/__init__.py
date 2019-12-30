from eval.npy2csv import get_csv, get_froc
from utils import env


def run_evaluation():
    args = env.get_args()
    if args.start_epoch == -1:
        args.start_epoch = 1
    get_csv(args)
    get_froc(args)
