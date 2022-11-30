import argparse
from pathlib import Path
import os
import time
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from data_loader import make_longtailed_imb, get_imbalanced, get_oversampled
from utils import InputNormalize, sum_t
from util.cutout import Cutout
from torchvision import datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0


def parse_args():
    model_options = ['resnet18', 'wideresnet']
    dataset_options = ['cifar10', 'cifar100']
    parser = argparse.ArgumentParser(description='Cutout Wide Residual Networks')
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--beta', default=0.999, type=float, help='Hyper-parameter for rejection/sampling')
    parser.add_argument('--stage2_epoch', default=40, type=int, help='Deferred strategy for re-balancing')
    parser.add_argument('--gamma', default=0.99, type=float, help='Threshold of the generation')
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup beta distribution parameters')
    parser.add_argument('--mixup', default=True, type=bool, help='stage1 whether to mixup')
    parser.add_argument('--eff_beta', default=1.0, type=float, help='Hyper-parameter for effective number')
    parser.add_argument('--focal_gamma', default=1.0, type=float, help='Hyper-parameter for Focal Loss')
    parser.add_argument('--step_size', default=0.1, type=float, help='')
    parser.add_argument('--attack_iter', default=10, type=int, help='')
    parser.add_argument('--imb_type', default='longtail', type=str,
                        choices=['none', 'longtail', 'step'],
                        help='Type of artificial imbalance')
    parser.add_argument('--ratio', default=100, type=int, help='max/min')
    parser.add_argument('--imb_start', default=5, type=int, help='start idx of step imbalance')
    parser.add_argument('--cost', '-c', action='store_true', help='oversampling')
    parser.add_argument('--effect_over', action='store_true', help='Use effective number in oversampling')
    parser.add_argument('--no_over', dest='over', action='store_false', help='Do not use over-sampling')

    parser.add_argument('--dataset', '-d', default='cifar100',
                        choices=dataset_options)
    parser.add_argument('--model', '-a', default='wideresnet',
                        choices=model_options)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=160,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='augment data by flipping and cropping')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0002, type=float)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--model_dir', default='ckps', type=str)
    parser.add_argument('--delta', default=0.9, type=float, help='Hyper-parameter for OPeN')
    parser.add_argument('--gama', default=0.5, type=float, help='Hyper-parameter for HoMu')
    parser.add_argument('--cfg',
                        default='./config/cifar100/cifar100_ir100_stage1_Mixup_len22.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('--resume',
                        default='./saved/cifar100_ir100_stage1_Mixup_len22_epoch160_202211032220/ckps/model_best.pth.tar',
                        help='experiment configure file name',
                        type=str)

    return parser.parse_args()


ARGS = parse_args()
DATASET = ARGS.dataset
BATCH_SIZE = ARGS.batch_size
MODEL = ARGS.model
DELTA = ARGS.delta
LR = ARGS.lr
EPOCH = ARGS.epochs
START_EPOCH = 0
print('==> Preparing data: %s' % DATASET)

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

mean = torch.tensor([0.4914, 0.4822, 0.4465])
std = torch.tensor([0.2023, 0.1994, 0.2010])
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
    Cutout(n_holes=ARGS.n_holes, length=ARGS.length)
    ])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if ARGS.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='/home/zeh/data',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='/home/zeh/data',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif ARGS.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)

if DATASET == 'cifar100':
    N_CLASSES = 100
    N_SAMPLES = 500
    mean = torch.tensor([0.5071, 0.4867, 0.4408])
    std = torch.tensor([0.2675, 0.2565, 0.2761])
elif DATASET == 'cifar10':
    N_CLASSES = 10
    N_SAMPLES = 5000
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
else:
    raise NotImplementedError()
#
normalizer = InputNormalize(mean, std).to(device)

# Data Loader
N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES)] * N_CLASSES
if ARGS.imb_type == 'longtail':
    N_SAMPLES_PER_CLASS_BASE = make_longtailed_imb(N_SAMPLES, N_CLASSES, ARGS.ratio)  # 构造长尾数据中每个类的样本个数
elif ARGS.imb_type == 'step':
    for i in range(ARGS.imb_start, N_CLASSES):
        N_SAMPLES_PER_CLASS_BASE[i] = int(N_SAMPLES * (1 / ARGS.ratio))

N_SAMPLES_PER_CLASS_BASE = tuple(N_SAMPLES_PER_CLASS_BASE)  # 将列表转化为元组，元组中的元素不能改变
print(N_SAMPLES_PER_CLASS_BASE)

# 获得不平衡数据，train_loader是不平衡的，val_loader和test_loader是平衡的数据集
imbalanced_train_loader, train_loader, val_loader, test_loader = get_imbalanced(DATASET, N_SAMPLES_PER_CLASS_BASE,
                                                                                BATCH_SIZE, train_transform,
                                                                                test_transform)
# To apply effective number for over-sampling or cost-sensitive
if ARGS.over and ARGS.effect_over:
    _beta = ARGS.eff_beta
    effective_num = 1.0 - np.power(_beta, N_SAMPLES_PER_CLASS_BASE)
    N_SAMPLES_PER_CLASS = tuple(np.array(effective_num) / (1 - _beta))
    print(N_SAMPLES_PER_CLASS)
else:
    N_SAMPLES_PER_CLASS = N_SAMPLES_PER_CLASS_BASE
N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS).to(device)  # 将列表转化为张量的形式


def evaluate(net, dataloader, epoch, fea_test_list, logger=None):
    is_training = net.training
    net.eval()

    total_loss = 0.0
    correct, total = 0.0, 0.0
    major1_correct, major2_correct, neutral_correct, minor1_correct, minor2_correct = 0.0, 0.0, 0.0, 0.0, 0.0
    major1_total, major2_total, neutral_total, minor1_total, minor2_total = 0.0, 0.0, 0.0, 0.0, 0.0

    class_correct = torch.zeros(N_CLASSES)
    class_total = torch.zeros(N_CLASSES)

    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs, fea_test = net(inputs)
            if epoch == 0:
                fea_test_list.append((fea_test, targets))
        predicted = torch.max(outputs.data, 1)[1]
        total += batch_size
        correct_mask = (predicted == targets)
        correct += sum_t(correct_mask)

        # For accuracy of minority / majority classes.
        major1_mask = targets < (N_CLASSES // 5)  # 前两类（0，1）记为大类major1_classes
        major1_total += sum_t(major1_mask)
        major1_correct += sum_t(correct_mask * major1_mask)

        major2_mask = ((N_CLASSES // 5) <= targets) & (targets < 2 * (N_CLASSES // 5))  # 第2,3类记为大类major2_classes
        major2_total += sum_t(major2_mask)
        major2_correct += sum_t(correct_mask * major2_mask)

        minor2_mask = targets >= (N_CLASSES - (N_CLASSES // 5))   # 最后两类（8，9）记为小类minor2_classes
        minor2_total += sum_t(minor2_mask)
        minor2_correct += sum_t(correct_mask * minor2_mask)

        # 第（6，7）记为小类minor1_classes
        minor1_mask = (3 * (N_CLASSES // 5) <= targets) & (targets < (N_CLASSES - (N_CLASSES // 5)))
        minor1_total += sum_t(minor1_mask)
        minor1_correct += sum_t(correct_mask * minor1_mask)

        # 既不属于major_classes也不属于minor_classes的类记为中型类即
        neutral_mask = ~(major1_mask + major2_mask + minor1_mask + minor2_mask)
        neutral_total += sum_t(neutral_mask)
        neutral_correct += sum_t(correct_mask * neutral_mask)

        for i in range(N_CLASSES):
            class_mask = (targets == i)
            class_total[i] += sum_t(class_mask)
            class_correct[i] += sum_t(correct_mask * class_mask)

    results = {
        'loss': total_loss / total,
        'acc': 100. * correct / total,
        'major1_acc': 100. * major1_correct / major1_total,
        'major2_acc': 100. * major2_correct / major2_total,
        'neutral_acc': 100. * neutral_correct / neutral_total,
        'minor1_acc': 100. * minor1_correct / minor1_total,
        'minor2_acc': 100. * minor2_correct / minor2_total,
        'class_acc': 100. * class_correct / class_total,
    }

    msg = 'Acc: %.3f%% | Major1_ACC: %.3f%% | Major2_ACC: %.3f%% |Neutral_ACC: %.3f%% | ' \
          'Minor1 ACC: %.3f%% | Minor2 ACC: %.3f%% ' % \
          (
              results['acc'], results['major1_acc'], results['major2_acc'], results['neutral_acc'],
              results['minor1_acc'], results['minor2_acc']
          )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    net.train(is_training)
    return results, fea_test


def create_logger(cfg_name):
    time_str = time.strftime('%Y%m%d%H%M')

    cfg_name = os.path.basename(cfg_name).split('.')[0]
    delta = 'delta{}'.format(ARGS.delta)
    gama = 'gama{}'.format(ARGS.gama)
    log_dir = Path("saved") / (cfg_name + '_' + delta + '_' + gama + '_' + time_str) / Path(ARGS.log_dir)
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.txt'.format(cfg_name)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir = Path("saved") / (cfg_name + '_' + delta + '_' + gama + '_' + time_str) / Path(ARGS.model_dir)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(model_dir)


def cor_class_create_logger(cfg_name):
    time_str = time.strftime('%Y%m%d%H%M')

    cfg_name = os.path.basename(cfg_name).split('.')[0]
    delta = 'delta{}'.format(ARGS.delta)
    log_dir = Path("saved") / (cfg_name + '_' + delta + '_' + time_str) / Path(ARGS.log_dir)
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.txt'.format(cfg_name)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir = Path("saved") / (cfg_name + '_corresponding_class_log') / Path(ARGS.model_dir)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(model_dir)


def create_logger_stage1(cfg_name):
    time_str = time.strftime('%Y%m%d%H%M')

    cfg_name = os.path.basename(cfg_name).split('.')[0]
    epoch = 'epoch{}'.format(ARGS.epochs)
    log_dir = Path("saved") / (cfg_name + '_' + epoch + '_' + time_str) / Path(ARGS.log_dir)
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.txt'.format(cfg_name)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    model_dir = Path("saved") / (cfg_name + '_' + epoch + '_' + time_str) / Path(ARGS.model_dir)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(model_dir)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    # Returns mixed inputs, pairs of targets, and lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)   # 贝塔分布
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()  # 输出为是把1到batch_size这些数随机打乱得到的一个数字序列
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_data_x(x, y, alpha=1.0, use_cuda=True):
    # Returns mixed inputs, pairs of targets, and lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)   # 贝塔分布
    else:
        lam = 1

    mixed_x = lam * x + (1 - lam) * x

    return mixed_x, y, y, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



# imagenet_dataset = torchvision.datasets.ImageFolder(
#     root='/home/zeh/Data/ImageNet/data/ImageNet2012/ILSVRC2012_img_train',
#     transform=data_transform)
# # 将数据集的标签和对应图片的位置加载为字典的形式
# imagenet_dataset_images = imagenet_dataset.imgs
# imagenet_dataset_targets = imagenet_dataset.targets
# imagenet_dataset_targets = {}.fromkeys(imagenet_dataset_targets).keys()
# UNIVERSUM_DATA_DICT = {i: [] for i in list(imagenet_dataset_targets)}
# for image in imagenet_dataset_images:
#     label = image[1]
#     UNIVERSUM_DATA_DICT[int(label)].append(image[0])
#
# tinyimagenet_dataset = torchvision.datasets.ImageFolder(
#     root='/home/zeh/Data/tiny-imagenet-200/train',
#     transform=data_transform)
# # 将数据集的标签和对应图片的位置加载为字典的形式
# tinyimagenet_dataset_images = tinyimagenet_dataset.imgs
# tinyimagenet_dataset_targets = tinyimagenet_dataset.targets
# tinyimagenet_dataset_targets = {}.fromkeys(tinyimagenet_dataset_targets).keys()
# UNIVERSUM_DATA_DICT_Tiny = {i: [] for i in list(tinyimagenet_dataset_targets)}
# for image in tinyimagenet_dataset_images:
#     label = image[1]
#     UNIVERSUM_DATA_DICT_Tiny[int(label)].append(image[0])

# universum_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     # GaussianBlur(kernel_size=int(0.1 * 32)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.4377, 0.4438, 0.4728],
#         std=[0.1201, 0.1231, 0.1052]),
#     Cutout(n_holes=ARGS.n_holes, length=ARGS.length)
#     ])
#
# universum_dataset = torchvision.datasets.SVHN(
#     root='/home/zeh/Data/SVHN',
#     split='train',
#     download=True,
#     transform=torchvision.transforms.ToTensor()
# )
# universum_loader = torch.utils.data.DataLoader(
#         universum_dataset, batch_size=256, shuffle=False, num_workers=0,
#         pin_memory=True)
# labels_num = list(set(universum_loader.dataset.labels))
# UNIVERSUM_DATA_DICT = {i: [] for i in labels_num}
# for datasets, labels in universum_loader:
#     for i in range(len(datasets)):
#         UNIVERSUM_DATA_DICT[int(labels[i])].append(datasets[i])



