import shutil
import torch.nn as nn
from tqdm import tqdm
from util.misc import CSVLogger
from model.resnet import ResNet18
from model.wide_resnet import WideResNet
from config import *

ARGS.cuda = not ARGS.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(ARGS.seed)
if ARGS.cuda:
    torch.cuda.manual_seed(ARGS.seed)

test_id = ARGS.dataset + '_' + ARGS.model

print(ARGS)

train_loader, train_instance_loader, test_loader = get_oversampled(DATASET, N_SAMPLES_PER_CLASS, BATCH_SIZE,
                                                                   train_transform, test_transform)
if ARGS.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif ARGS.model == 'wideresnet':
    if ARGS.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=ARGS.lr, momentum=0.9,
                                nesterov=True, weight_decay=ARGS.weight_decay)

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=ARGS, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


def save_checkpoint(state, is_best, model_dir):
    file_name = model_dir + '/current.pth.tar'
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, model_dir + '/model_best.pth.tar')


logger, model_dir = create_logger_stage1(ARGS.cfg)


def adjust_learning_rate(optimizer, lr_init, epoch):
    """decrease the learning rate at 160 and 180 epoch ( from LDAM-DRW, NeurIPS19 )"""
    lr = lr_init

    if epoch < 5:  # 前五个epoch采用‘linear warm-up’的学习策略
        lr = (epoch + 1) * lr_init / 5

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(ARGS.epochs):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    accuracy = 0.
    best_acc1 = 0.

    progress_bar = tqdm(train_instance_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        adjust_learning_rate(cnn_optimizer, LR, epoch)

        cnn.zero_grad()
        if ARGS.mixup is True:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=ARGS.alpha)
            pred = cnn(images)
            xentropy_loss = mixup_criterion(criterion, pred, targets_a, targets_b, lam)
        else:
            pred = cnn(images)
            xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()
        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
    # remember best acc@1 and save checkpoint
    is_best = accuracy > best_acc1
    best_acc1 = max(accuracy, best_acc1)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict_model': cnn.state_dict(),
        'best_acc1': best_acc1,
    }, is_best, model_dir)

    test_acc = test(test_loader)
    logger.info('* epoch {epoch}  Acc@1 {top1:.3f}  test_acc {test_acc:.3f}'.format(epoch=epoch+1, top1=best_acc1, test_acc=test_acc))

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
