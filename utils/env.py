import argparse
import os

def num_arr(value):
    return [float(i) for i in value.split(',') if i != '']

parser = argparse.ArgumentParser(description='LNDCML')
parser.add_argument('--job', metavar='JOB', default='train', help='prepare/train')
parser.add_argument('--model', '-m', metavar='MODEL', default='res18', help='模型名称 res18 / dpn3d26')
parser.add_argument('--id', metavar='MODEL_ID', default='noid', help='这次运行的模型ID，主要用于区分超参数')
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default=1, type=int, metavar='N',
                    help='是否继续上一次的模型，不继续会删除之前的记录')
parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                    help='从上一次的哪个epoch开始，-1代表最新')
parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshold for get pbb')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--gpu-test', default=4, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--nd-train', default=9, type=int, metavar='N',
                    help='用多少个文件夹训练')
parser.add_argument('--nd-test', default=1, type=int, metavar='N',
                    help='用多少个文件夹测试')
parser.add_argument('--multi-process', default=0, type=int, metavar='N',
                    help='是否多线程执行 可用阶段 eval')
parser.add_argument('--eval-detp', default=[-1, -1.5], type=num_arr, metavar='N',
                    help='eval所用的detp')
parser.add_argument('--cls-train-file-size', default=-1, type=int, metavar='N',
                    help='cls的train用多少个文件，-1为全部')




def get_args():
    return parser.parse_args()


def get(key: str) -> str:
    val = os.getenv(key)
    if val is None:
        raise ValueError('Env value %s is not set' % key)
    return val
