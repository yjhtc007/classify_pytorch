# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from PIL import Image
from sklearn.model_selection import KFold

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The dir of train data
pic_dir = '/home/xxx/DATA/'

# input dataset
train_jpg = np.array(glob.glob(pic_dir+'*/*.png'))

# create the label map dict
labels = {}
for i, file in enumerate(os.listdir(pic_dir)):
    labels.update({file:i})

# custom dataset class
class myDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        # transform the data or not
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.train_jpg[index]).convert('RGB')
        max_length = max(img.size[0], img.size[1])
        bg_img = Image.new('RGB', (max_length, max_length), color=(0,0,0))
        bg_img.paste(img, (0,(max(img.size[0], img.size[1])-min(img.size[0], img.size[1]))//2,max(img.size[0], img.size[1]),(max(img.size[0], img.size[1])+min(img.size[0], img.size[1]))//2))
        img = bg_img
        if self.transform is not None:
            img = self.transform(img)
        label_num = self.train_jpg[index]
        # replace the '\\' to '/' in the file path
        if '\\' in label_num:
            label_num = label_num.replace('\\', '/')

        return img, torch.from_numpy(np.array(labels[label_num.split('/')[-2]]))

    def __len__(self):
        return len(self.train_jpg)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class VGGNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGNetwork, self).__init__()

        # choice the VGG16
        model = models.vgg16_bn(True)   # load the pretrain model
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Sequential(nn.Linear(512, 4096),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(4096, 4096),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(4096,num_classes))

        self.net = model

    def forward(self, img):
        out = self.net(img)
        return out


class ResNet34Net(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet34Net, self).__init__()

        # choice the ResNet34
        model = models.resnet34(True)   # load the pretrain model
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, num_classes)

        self.net = model

    def forward(self, img):
        out = self.net(img)
        return out


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        return top1


def train(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target.long())

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target.long(), topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)

# ------ config ------
# use 10-fold cross-validation
skf = KFold(n_splits=10, random_state=2021, shuffle=True)
# numbers of classes
NUM_CLASSES = 2
# learning rate
LEARNING_RATE = 0.01
# batch size
BATCH_SIZE = 16
MODEL_NAME = 'ResNet34'
# save dir
SAVE_DIR = './model'


for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg, train_jpg)):
    print('current_flod_idx:', flod_idx)
    train_loader = torch.utils.data.DataLoader(
        myDataset(train_jpg[train_idx],
                  transforms.Compose([
                      # transforms.Pad(PADDING_SIZE, fill=255), # 填充
                      # transforms.RandomGrayscale(),   # 依概率p转为灰度图
                      transforms.Resize((224, 224)),    # resize
                      # transforms.RandomAffine(10),    # 仿射变换
                      # transforms.ColorJitter(hue=.05, saturation=.05),    # 修改亮度、对比度和饱和度
                      # transforms.RandomCrop((450, 450)),  # 随机裁剪
                      # transforms.RandomHorizontalFlip(),  # 依概率p水平翻转
                      # transforms.RandomVerticalFlip(),    # 依概率p垂直翻转
                      transforms.ToTensor(),    # 转为tensor，并归一化至[0-1]
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # 标准化
                  ])
                  ), batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        myDataset(train_jpg[val_idx],
                  transforms.Compose([
                      transforms.Resize((224, 224)),
                      # transforms.Resize((124, 124)),
                      # transforms.RandomCrop((450, 450)),
                      # transforms.RandomCrop((88, 88)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=16, shuffle=False, num_workers=0, pin_memory=True
    )
    if MODEL_NAME == 'VGG':
        model = VGGNetwork().to(device)
    elif MODEL_NAME == 'ResNet34':
        model = ResNet34Net().to(device)
    else:
        raise ValueError('the input model name is invalid')
    # model = nn.DataParallel(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
    best_acc = 0.0
    for epoch in range(30):
        print('The Current Epoch: ', epoch)
        train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)

        scheduler.step()

        if val_acc.avg.item() > best_acc:
            best_acc = val_acc.avg.item()
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_accuracy.pth'))

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, '{}_fold{}_{}.pth'.format(MODEL_NAME, str(epoch), str(flod_idx))))

    # break
