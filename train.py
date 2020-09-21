import logging
import sys

import torch
import numpy as np
import cv2
import argparse
#活体检测数据集--四张图片--送入网络的是差分图，法线图，反射率图
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from  vision.ssd.data_preprocessing import TestTransform,TrainAugmentation
from vision.EfficientNet import efficient_net
parser = argparse.ArgumentParser(description='Convolutional Neural Network for Live Face Detection')
parser.add_argument('--dataset_type',default='voc',type=str)
parser.add_argument('--dataset',type=str,default='',help='Dataset directory path')
parser.add_argument('--net',default='efficient_net',type=str)
parser.add_argument('--lr',default=1e-4,type=float,help='initial learning rate')
parser.add_argument('--momentum',default=0.9,type=float,help='Momentum value for optim')
parser.add_argument('--weight_decay',default=5e-4,type=float,help='Weight decay for SGD')
parser.add_argument('--scheduler',default='cosine',type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")
parser.add_argument('--milestones',default='80,100',type=str,help="milestones for MultiStepLR")

parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train_2202 model')
args = parser.parse_args()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu')

# logging.basicConfig函数对日志的输出格式及方式做相关配置
#默认生成的root logger的level是logging.WARNING, 低于该级别的就不输出了
#级别排序:CRITICAL > ERROR > WARNING > INFO > DEBUG
#下面代码通过logging.basicConfig函数进行配置了日志级别和日志内容输出格式；
# 因为级别为DEBUG，所以会将DEBUG级别以上的信息都输出显示再控制台上。
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


# 如果网络的输入数据维度或类型上变化不大，设置torch.backends.cudnn.benchmark = true可以增加运行效率；
# 如果网络的输入数据在每次iteration都变化的话，会导致cnDNN每次都会去寻找一遍最优配置，这样反而会降低运行效率。
if torch.cuda.is_available() and args.use_cuda:
    torch.backends.cudnn.benchmark = True
    logging.info('use cuda.')

if __name__ == '__main__':
    logging.info(args)
    #创建网络
    if args.net == 'efficient_net':
        efficientNet = efficient_net()
        #config里存储一些基础信息：image_size
        config = 2
    elif args.net == 'res_net50':
        create_net = efficient_net()
        config = 2
    else:
        logging.fatal("The net type is wrong.")
        #?
        parser.print_help(sys.stderr)
        sys.exit(1)
    #加载数据预处理方法--这里是ssd的数据预处理方法--难点
    train_transform = TrainAugmentation()
    #label_transform
    #test_transform = TestTransform()
    #####################

    logging.info("Prepare training datasets.")