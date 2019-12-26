from eval.npy2csv import get_csv, get_froc
from utils import env


def run_evaluation():
    args = env.get_args()
    detp = [-2]
    get_csv(detp, args)
    get_froc(detp, args)
