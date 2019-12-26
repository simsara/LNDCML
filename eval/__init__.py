from eval.npy2csv import get_csv, get_froc
from utils import env

uid_label = 'uid'
x_label = 'coord_x'
y_label = 'coord_y'
z_label = 'coord_z'
diameter_mm_label = 'diameter_mm'
probability_label = 'probability'

pbb_csv_header = [uid_label, x_label, y_label, z_label, probability_label]

def run_evaluation():
    args = env.get_args()
    detp = [-2]
    get_csv(detp, args)
    get_froc(detp, args)
