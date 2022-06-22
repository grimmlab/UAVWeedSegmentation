# This implementation is for Resnets without the need of the basic blocks
# Thus is easier for re-implementation and changes

from turtle import down
import torch
import torch.nn as nn
from collections import OrderedDict
from torchinfo import summary
from torchvision._internally_replaced_utils import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    }

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, # TODO: check if that is always 1
                     padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, padding=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=3, replace_stride_with_dilation=False):
        super().__init__()
        self.num_classes = num_classes
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #first basic block
        layers = [BasicBlock(in_ch=64, out_ch=64, stride=1, downsample=None),
                  BasicBlock(in_ch=64, out_ch=64, stride=1, downsample=None)
                  ]
        self.layer1 = nn.Sequential(*layers)
        #second basic block, here we need also a "downsample" sequential
        layers = [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(128)
                  ]
        downsample = nn.Sequential(*layers)
        
        layers = [BasicBlock(in_ch=64, out_ch=128, stride=2, downsample=downsample),
                  BasicBlock(in_ch=128, out_ch=128, stride=1, downsample=None)]

        self.layer2 = nn.Sequential(*layers)

        # third basic block, here we need also a "downsample" sequential
        
        if self.replace_stride_with_dilation:
            layers = [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(256)
                  ]# TODO
        else:
            layers = [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(256)
                  ]

        downsample = nn.Sequential(*layers)
        if self.replace_stride_with_dilation:
            layers = [BasicBlock(in_ch=128, out_ch=256, stride=2, downsample=downsample),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None)]# TODO
        else:
            layers = [BasicBlock(in_ch=128, out_ch=256, stride=2, downsample=downsample),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None)]

        self.layer3 = nn.Sequential(*layers)

        # fourth basic block, here we need also a "downsample" sequential
        if self.replace_stride_with_dilation:
            layers = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(512)
                  ]# TODO
        else:
            layers = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(512)
                  ]
        downsample = nn.Sequential(*layers)
        if self.replace_stride_with_dilation:
            layers = [BasicBlock(in_ch=256, out_ch=512, stride=2, downsample=downsample),
                  BasicBlock(in_ch=512, out_ch=512, stride=1, downsample=None)]# TODO
        else:
            layers = [BasicBlock(in_ch=256, out_ch=512, stride=2, downsample=downsample),
                  BasicBlock(in_ch=512, out_ch=512, stride=1, downsample=None)]

        self.layer4 = nn.Sequential(*layers)

    def forward(self, x):
        layers = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        layers["layer1"]=x
        x = self.layer2(x)
        layers["layer2"]=x
        x = self.layer3(x)
        layers["layer3"]=x
        x = self.layer4(x)
        layers["layer4"]=x

        return layers

class ResNet34(nn.Module):
    def __init__(self, num_classes=3, replace_stride_with_dilation=False):
        super().__init__()
        self.num_classes = num_classes
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #first basic block
        layers = [BasicBlock(in_ch=64, out_ch=64, stride=1, downsample=None),
                  BasicBlock(in_ch=64, out_ch=64, stride=1, downsample=None),
                  BasicBlock(in_ch=64, out_ch=64, stride=1, downsample=None)
                  ]
        self.layer1 = nn.Sequential(*layers)
        #second basic block, here we need also a "downsample" sequential
        layers = [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(128)
                  ]
        downsample = nn.Sequential(*layers)
        
        layers = [BasicBlock(in_ch=64, out_ch=128, stride=2, downsample=downsample),
                  BasicBlock(in_ch=128, out_ch=128, stride=1, downsample=None),
                  BasicBlock(in_ch=128, out_ch=128, stride=1, downsample=None),
                  BasicBlock(in_ch=128, out_ch=128, stride=1, downsample=None)]

        self.layer2 = nn.Sequential(*layers)

        # third basic block, here we need also a "downsample" sequential
        if self.replace_stride_with_dilation:
            layers = [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(256)
                  ]# TODO
        else:
            layers = [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(256)
                  ]
        downsample = nn.Sequential(*layers)
        
        if self.replace_stride_with_dilation:
            layers = [BasicBlock(in_ch=128, out_ch=256, stride=2, downsample=downsample),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None)]# TODO
        else:
            layers = [BasicBlock(in_ch=128, out_ch=256, stride=2, downsample=downsample),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=1, downsample=None)]

        self.layer3 = nn.Sequential(*layers)

        # fourth basic block, here we need also a "downsample" sequential
        if self.replace_stride_with_dilation:
            layers = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(512)
                  ]# TODO
        else:
            layers = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(512)
                  ]
        downsample = nn.Sequential(*layers)
        
        if self.replace_stride_with_dilation:
            layers = [BasicBlock(in_ch=256, out_ch=512, stride=2, downsample=downsample),
                  BasicBlock(in_ch=512, out_ch=512, stride=1, downsample=None),
                  BasicBlock(in_ch=512, out_ch=512, stride=1, downsample=None)]# TODO
        else:
            layers = [BasicBlock(in_ch=256, out_ch=512, stride=2, downsample=downsample),
                  BasicBlock(in_ch=512, out_ch=512, stride=1, downsample=None),
                  BasicBlock(in_ch=512, out_ch=512, stride=1, downsample=None)]

        self.layer4 = nn.Sequential(*layers)

    def forward(self, x):
        layers = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        layers["layer1"]=x
        x = self.layer2(x)
        layers["layer2"]=x
        x = self.layer3(x)
        layers["layer3"]=x
        x = self.layer4(x)
        layers["layer4"]=x

        return layers

class ResNet50(nn.Module):
    def __init__(self, num_classes=3, replace_stride_with_dilation=False):
        super().__init__()
        self.num_classes = num_classes
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        #first Bottleneck block
        layers = [nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, bias=False), 
                  nn.BatchNorm2d(256)
                  ]
        downsample = nn.Sequential(*layers)

        layers = [Bottleneck(in_ch=64, mid_ch=64, out_ch=256, stride=1, downsample=downsample),
                  Bottleneck(in_ch=256, mid_ch=64, out_ch=256, stride=1, downsample=None),
                  Bottleneck(in_ch=256, mid_ch=64, out_ch=256, stride=1, downsample=None)
                  ]
        self.layer1 = nn.Sequential(*layers)


        #second Bottleneck block
        layers = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(512)
                  ]
        downsample = nn.Sequential(*layers)

        layers = [Bottleneck(in_ch=256, mid_ch=128, out_ch=512, stride=2, downsample=downsample),
                  Bottleneck(in_ch=512, mid_ch=128, out_ch=512, stride=1, downsample=None),
                  Bottleneck(in_ch=512, mid_ch=128, out_ch=512, stride=1, downsample=None),
                  Bottleneck(in_ch=512, mid_ch=128, out_ch=512, stride=1, downsample=None),
                  ]
        self.layer2 = nn.Sequential(*layers)

        # third Bottleneck block
        if self.replace_stride_with_dilation:
            layers = [nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, bias=False), 
                  nn.BatchNorm2d(1024)
                  ]
        else:
            layers = [nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(1024)
                  ]
        downsample = nn.Sequential(*layers)
        
        if self.replace_stride_with_dilation:
            layers = [Bottleneck(in_ch=512, mid_ch=256, out_ch=1024, stride=1, downsample=downsample),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, padding=2, dilation=2, downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, padding=2, dilation=2,downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, padding=2, dilation=2,downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, padding=2, dilation=2,downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, padding=2, dilation=2,downsample=None),
                  ]
        else:
            layers = [Bottleneck(in_ch=512, mid_ch=256, out_ch=1024, stride=2, downsample=downsample),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=1, downsample=None),
                  ]

        self.layer3 = nn.Sequential(*layers)

        # fourth Bottleneck block
        if self.replace_stride_with_dilation:
            layers = [nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, bias=False), 
                  nn.BatchNorm2d(2048)
                  ]
        else:
            layers = [nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(2048)
                  ]
        downsample = nn.Sequential(*layers)
        if self.replace_stride_with_dilation:
            layers = [Bottleneck(in_ch=1024, mid_ch=512, out_ch=2048, stride=1, padding=2, dilation=2, downsample=downsample),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=1, padding=4, dilation=4, downsample=None),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=1, padding=4, dilation=4, downsample=None),
                  ]
        else:
            layers = [Bottleneck(in_ch=1024, mid_ch=512, out_ch=2048, stride=2, downsample=downsample),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=1, downsample=None),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=1, downsample=None),
                  ]

        self.layer4 = nn.Sequential(*layers)

    def forward(self, x):
        layers = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        layers["layer1"]=x
        x = self.layer2(x)
        layers["layer2"]=x
        x = self.layer3(x)
        layers["layer3"]=x
        x = self.layer4(x)
        layers["layer4"]=x

        return layers


def load_resnet(encoder_name, num_classes, pretrained, replace_stride_with_dilation, progress=True):
    if encoder_name == "resnet18":
        model = ResNet18(num_classes=num_classes, replace_stride_with_dilation=replace_stride_with_dilation)
    elif encoder_name == "resnet34":
        model = ResNet34(num_classes=num_classes, replace_stride_with_dilation=replace_stride_with_dilation)
    elif encoder_name == "resnet50":
        model = ResNet50(num_classes=num_classes, replace_stride_with_dilation=replace_stride_with_dilation)
    elif encoder_name == "resnet101":
        raise NotImplementedError(f"{encoder_name} is not implemented.")
    elif encoder_name == "resnet152":
        raise NotImplementedError(f"{encoder_name} is not implemented.")
    else:
        raise NotImplementedError(f"{encoder_name} is not implemented.")

    if pretrained:
        print("LOADING PRETRAINED MODEL WEIGHTS FROM IMAGENET")
        # get the state dict from URL
        state_dict = load_state_dict_from_url(model_urls[encoder_name],
                                              progress=progress)
        # we need to remove the keys for the fully connected layer, as we only need the feature extractor
        entries_to_remove = ('fc.weight', 'fc.bias')
        for k in entries_to_remove:
            state_dict.pop(k, None)
        # actually loading the weights to the model    
        model.load_state_dict(state_dict) 
    else:
        print("TRAINING WITH RANDOM INITIALIZED WEIGHTS")
    return model


def test():
    x = torch.randn(2, 3, 512, 512).to("cuda")
    for encoder_name in ["resnet50"]:
        rn18 = load_resnet(encoder_name=encoder_name, num_classes=3, pretrained=True, replace_stride_with_dilation=True).to("cuda")
        rn18_preds = rn18(x).to('cuda')
        #summary(rn18, x.shape)
        #print(f"{encoder_name}- {rn18_preds.shape=}")

if __name__ == "__main__":
    test()