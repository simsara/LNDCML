import detect
import prepare
from utils import env

if __name__ == '__main__':
    args = env.get_args()
    if args.job == 'prepare':
        prepare.prepare_luna()
    elif args.job == 'train':
        detect.run_train()
    else:
        raise ValueError('Not supported job name [%s]' % args.job)
