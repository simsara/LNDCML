import detect
import detect_test
import nodcls
import prepare
import eval as evaluation
from utils import env

if __name__ == '__main__':
    args = env.get_args()
    if args.job == 'prepare':
        prepare.prepare_luna()
    elif args.job == 'train':
        detect.run_train()
    elif args.job == 'test':
        detect.run_test()
    elif args.job == 'val':
        detect.run_validate()
    elif args.job == 'eval':
        evaluation.run_evaluation()
    elif args.job == 'cls-prepare':
        nodcls.preprocess()
    elif args.job == 'cls-train':
        nodcls.run_train()
    elif args.job == 'detect-prod':
        detect_test.run()
    elif args.job == 'cls-run-gbm':
        nodcls.run_gbm()
    elif args.job == 'cls-sk':
        nodcls.cls_with_net()
    else:
        raise ValueError('Not supported job name [%s]' % args.job)
