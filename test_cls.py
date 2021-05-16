# -*- coding: utf-8 -*-
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# ------ config ------
RE_SIZE = 224
MEAN_R, MEAN_G, MEAN_B = 0.485, 0.456, 0.406
STD_R, STD_G, STD_B = 0.229, 0.224, 0.225
NUM_CLASSES = 2
# the train data dir, as the same with the training process
TRAIN_DATA_DIR = '/home/xxx/DATA/'
# the test pic name
TEST_DATA_DIR = './xxx.png'
# the model dir
CHECKPOINTS_ROOT = './model/'
# the model name
LOAD_MODEL_PATH = 'xxx.pth'
# the Network type
MODEL_NAME = 'ResNet34'

labels = {}
idx_to_class = {}
for i, file in enumerate(os.listdir(TRAIN_DATA_DIR)):
    labels.update({file:i})
    idx_to_class.update({i:file})

transform = transforms.Compose([
    transforms.Resize((RE_SIZE, RE_SIZE)),
    # transforms.Pad(PADDING_SIZE, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN_R, MEAN_G, MEAN_B],
                         std = [STD_R, STD_G, STD_B])
])


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


def predict_single(pic_path, idx_to_class=None, num_class=None, CHECKPOINTS_ROOT=None,  LOAD_MODEL_PATH=None):
    if MODEL_NAME == 'ResNet34':
        model = ResNet34Net(num_classes=num_class).eval()
    elif MODEL_NAME == 'VGG16':
        model = VGGNetwork(num_classes=num_class).eval()
    else:
        raise ValueError('the input model name is invalid')
    if LOAD_MODEL_PATH:
        model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_ROOT, LOAD_MODEL_PATH)))
        print('load model from {}'.format(os.path.join(CHECKPOINTS_ROOT, LOAD_MODEL_PATH)))
    pic = Image.open(pic_path).convert('RGB')
    pic = torch.unsqueeze(transform(pic),dim=0)
    output = model(pic)

    result = idx_to_class[torch.max(output, 1)[1].item()]

    return result


if __name__ == '__main__':
    result = predict_single(pic_path=TEST_DATA_DIR, idx_to_class=idx_to_class, num_class=NUM_CLASSES, CHECKPOINTS_ROOT=CHECKPOINTS_ROOT, LOAD_MODEL_PATH=LOAD_MODEL_PATH)
    print('The predict result is:', result)
