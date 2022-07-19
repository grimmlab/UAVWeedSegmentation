# This implementation is for Resnets without the need of the basic blocks
# Thus is easier for re-implementation and changes
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision._internally_replaced_utils import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    }

def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, _ = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,
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
    def __init__(self, num_classes=3, output_stride=32):
        super().__init__()
        self.num_classes = num_classes
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

        layers = [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(256)
                  ]

        downsample = nn.Sequential(*layers)
        strides = [2, 1]
        paddings = [1, 1]
        dilations = [1, 1]

        layers = [BasicBlock(in_ch=128, out_ch=256, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  BasicBlock(in_ch=256, out_ch=256, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None)]

        self.layer3 = nn.Sequential(*layers)

        # fourth basic block, here we need also a "downsample" sequential
        layers = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(512)
                  ]
        downsample = nn.Sequential(*layers)
        strides = [2, 1]
        paddings = [1, 1]
        dilations = [1, 1]

        layers = [BasicBlock(in_ch=256, out_ch=512, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  BasicBlock(in_ch=512, out_ch=512, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None)]

        self.layer4 = nn.Sequential(*layers)
        if output_stride !=32:
            self.make_dilated(output_stride=output_stride)

    def forward(self, x):
        layers = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        layers["layer0"]=x
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

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]
    
    def make_dilated(self, output_stride):
        print(f"making dilated model")
        if output_stride == 16:
            stage_list=[5,]
            dilation_list=[2,]
            
        elif output_stride == 8:
            stage_list=[4, 5]
            dilation_list=[2, 4] 

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))
        
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

class ResNet34(nn.Module):
    def __init__(self, num_classes=3, output_stride=32):
        super().__init__()
        self.num_classes = num_classes
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
        layers = [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(256)
                  ]
        downsample = nn.Sequential(*layers)
        
        strides = [2, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1]
        dilations = [1, 1, 1, 1, 1, 1]

        layers = [BasicBlock(in_ch=128, out_ch=256, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  BasicBlock(in_ch=256, out_ch=256, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=strides[2], padding=paddings[2], dilation=dilations[2], downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=strides[3], padding=paddings[3], dilation=dilations[3], downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=strides[4], padding=paddings[4], dilation=dilations[4], downsample=None),
                  BasicBlock(in_ch=256, out_ch=256, stride=strides[5], padding=paddings[5], dilation=dilations[5], downsample=None)]

        self.layer3 = nn.Sequential(*layers)

        # fourth basic block, here we need also a "downsample" sequential
        layers = [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(512)
                  ]
        downsample = nn.Sequential(*layers)
        

        strides = [2, 1, 1]
        paddings = [1, 1, 1]
        dilations = [1, 1, 1]

        layers = [BasicBlock(in_ch=256, out_ch=512, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  BasicBlock(in_ch=512, out_ch=512, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None),
                  BasicBlock(in_ch=512, out_ch=512, stride=strides[2], padding=paddings[2], dilation=dilations[2], downsample=None)]

        self.layer4 = nn.Sequential(*layers)
        if output_stride !=32:
            self.make_dilated(output_stride=output_stride)

    def forward(self, x):
        layers = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        layers["layer0"]=x
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

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]
    
    def make_dilated(self, output_stride):
        print(f"making dilated model")
        if output_stride == 16:
            stage_list=[5,]
            dilation_list=[2,]
            
        elif output_stride == 8:
            stage_list=[4, 5]
            dilation_list=[2, 4] 

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))
        
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

class ResNet50(nn.Module):
    """
    check if we can simplify the replace_stride_with_dilation if statement
    """
    def __init__(self, num_classes=3, output_stride=32):
        super().__init__()
        self.num_classes = num_classes
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
        layers = [nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(1024)
                  ]
        
        downsample = nn.Sequential(*layers)
        strides = [2, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1]
        dilations = [1, 1, 1, 1, 1, 1]


        layers = [Bottleneck(in_ch=512, mid_ch=256, out_ch=1024, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[2], padding=paddings[2], dilation=dilations[2], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[3], padding=paddings[3], dilation=dilations[3], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[4], padding=paddings[4], dilation=dilations[4], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[5], padding=paddings[5], dilation=dilations[5], downsample=None),
                  ]

        self.layer3 = nn.Sequential(*layers)

        # fourth Bottleneck block

        layers = [nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(2048)
                  ]

        downsample = nn.Sequential(*layers)

        strides = [2, 1, 1]
        paddings = [1, 1, 1]
        dilations = [1, 1, 1]

        layers = [Bottleneck(in_ch=1024, mid_ch=512, out_ch=2048, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=strides[2], padding=paddings[2], dilation=dilations[2], downsample=None),
                  ]

        self.layer4 = nn.Sequential(*layers)
        if output_stride !=32:
            self.make_dilated(output_stride=output_stride)

    def forward(self, x):
        layers = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        layers["layer0"]=x
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

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]
    
    def make_dilated(self, output_stride):
        print(f"making dilated model")
        if output_stride == 16:
            stage_list=[5,]
            dilation_list=[2,]
            
        elif output_stride == 8:
            stage_list=[4, 5]
            dilation_list=[2, 4] 

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))
        
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )


class ResNet101(nn.Module):
    """
    check if we can simplify the replace_stride_with_dilation if statement
    """
    def __init__(self, num_classes=3, output_stride=32):
        super().__init__()
        self.num_classes = num_classes
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
        layers = [nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(1024)
                  ]
        
        downsample = nn.Sequential(*layers)
        strides = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


        layers = [Bottleneck(in_ch=512, mid_ch=256, out_ch=1024, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[2], padding=paddings[2], dilation=dilations[2], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[3], padding=paddings[3], dilation=dilations[3], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[4], padding=paddings[4], dilation=dilations[4], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[5], padding=paddings[5], dilation=dilations[5], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[6], padding=paddings[6], dilation=dilations[6], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[7], padding=paddings[7], dilation=dilations[7], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[8], padding=paddings[8], dilation=dilations[8], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[9], padding=paddings[9], dilation=dilations[9], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[10], padding=paddings[10], dilation=dilations[10], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[11], padding=paddings[11], dilation=dilations[11], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[12], padding=paddings[12], dilation=dilations[12], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[13], padding=paddings[13], dilation=dilations[13], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[14], padding=paddings[14], dilation=dilations[14], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[15], padding=paddings[15], dilation=dilations[15], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[16], padding=paddings[16], dilation=dilations[16], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[17], padding=paddings[17], dilation=dilations[17], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[18], padding=paddings[18], dilation=dilations[18], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[19], padding=paddings[19], dilation=dilations[19], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[20], padding=paddings[20], dilation=dilations[20], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[21], padding=paddings[21], dilation=dilations[21], downsample=None),
                  Bottleneck(in_ch=1024, mid_ch=256, out_ch=1024, stride=strides[22], padding=paddings[22], dilation=dilations[22], downsample=None),
                  ]

        self.layer3 = nn.Sequential(*layers)

        # fourth Bottleneck block
        layers = [nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, bias=False), 
                  nn.BatchNorm2d(2048)
                  ]

        downsample = nn.Sequential(*layers)

        strides = [2, 1, 1]
        paddings = [1, 1, 1]
        dilations = [1, 1, 1]

        layers = [Bottleneck(in_ch=1024, mid_ch=512, out_ch=2048, stride=strides[0], padding=paddings[0], dilation=dilations[0], downsample=downsample),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=strides[1], padding=paddings[1], dilation=dilations[1], downsample=None),
                  Bottleneck(in_ch=2048, mid_ch=512, out_ch=2048, stride=strides[2], padding=paddings[2], dilation=dilations[2], downsample=None),
                  ]

        self.layer4 = nn.Sequential(*layers)
        if output_stride !=32:
            self.make_dilated(output_stride=output_stride)

    def forward(self, x):
        layers = OrderedDict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        layers["layer0"]=x
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

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]
    
    def make_dilated(self, output_stride):
        print(f"making dilated model")
        if output_stride == 16:
            stage_list=[5,]
            dilation_list=[2,]
            
        elif output_stride == 8:
            stage_list=[4, 5]
            dilation_list=[2, 4] 

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))
        
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )



def load_resnet(encoder_name, num_classes, pretrained, replace_stride_with_dilation, progress=True):
    if replace_stride_with_dilation:
        print(f"replacing stride with dilation")
        output_stride = 8
    else:
        output_stride = 32
    if encoder_name == "resnet18":
        model = ResNet18(num_classes=num_classes, output_stride=output_stride)
        for param in model.parameters():
            param.requires_grad = True
    elif encoder_name == "resnet34":
        model = ResNet34(num_classes=num_classes, output_stride=output_stride)
        for param in model.parameters():
            param.requires_grad = True
    elif encoder_name == "resnet50":
        model = ResNet50(num_classes=num_classes, output_stride=output_stride)
        for param in model.parameters():
            param.requires_grad = True
    elif encoder_name == "resnet101":
        model = ResNet101(num_classes=num_classes, output_stride=output_stride)
        for param in model.parameters():
            param.requires_grad = True
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
    x = torch.randn(20, 3, 256, 256).to("cuda")
    for encoder_name in ["resnet34"]:
        for output_stride in [True, False]:
            model = load_resnet(encoder_name=encoder_name, num_classes=3, pretrained=True, replace_stride_with_dilation=output_stride).to("cuda")
            rn18_preds = model(x)["layer4"]

if __name__ == "__main__":
    test()