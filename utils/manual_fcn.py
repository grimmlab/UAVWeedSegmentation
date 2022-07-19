
import torch
from torch import nn
from torch.nn import functional as F
from .manual_resnet import load_resnet


class FCNHead(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        """
        Caution: we still use Dropout, even tho not in the original implementation
        """
        mid_ch = in_ch // 4
        layers = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        ]

        super().__init__(*layers)

class FCNtv(nn.Module):
    """
    FCN network without skip connections. 
    If there is no dilation in the backbone, the model will be equal to FCN32s.
    The interpolation will make [B, C, 16, 16] --> [B, C, 256, 256]
    If a dilation of x4 is used in the backbone, 
    then the model will be equal to FCN8s 
    without any skip connections between intermediate layers
    The interpolation will make [B, C, 64, 64] --> [B, C, 256, 256]
    Currently implemented is only bilinear interpolation. 
    """
    def __init__(self, encoder_name, backbone, head, num_classes=3, n_upsample=32, b_bilinear=True, replace_stride_with_dilation=False):
        super().__init__()
        self.encoder_name = "resnet50"
        self.backbone = backbone
        self.head = head
        print(f"using test {encoder_name}, {n_upsample}x upsampling with replace strides {replace_stride_with_dilation} and bilinear {b_bilinear}")


    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)["layer4"]
        x = self.head(x)

        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x


class FCNskip(nn.Module):

    def __init__(self, encoder_name, backbone, head, num_classes=3, n_upsample=32, b_bilinear=True, replace_stride_with_dilation=False):
        super().__init__()
        self.n_upsample = n_upsample
        self.encoder_name = encoder_name
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.b_bilinear=b_bilinear
        print(f"using {encoder_name}, {n_upsample}x upsampling with replace strides {replace_stride_with_dilation} and bilinear {b_bilinear}")

        self.bn = nn.BatchNorm2d(num_features=num_classes)
        self.onebyone128 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.onebyone256 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.onebyone512 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.onebyone1024 = nn.Conv2d(1024, num_classes, kernel_size=1)
        if not b_bilinear:
            self.convTranspose = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.backbone = backbone


        self.head = head

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)

        layer4 = self.head(x["layer4"])

        if self.n_upsample == 16: # use FCN-16s model
            # upsampling needs to be based on the num_classes feature maps

            # get intermediate layer and match channels
            if self.encoder_name in ["resnet18", "resnet34"]:
                layer3 = self.onebyone256(x["layer3"])
            else:
                layer3 = self.onebyone1024(x["layer3"])
            # upsample last layer 2x to match spatial resolution
            if self.b_bilinear:
                x = F.interpolate(layer4, scale_factor=2.0, mode="bilinear", align_corners=False)
            else:
                x = self.convTranspose(layer4)
            # concat both
            x = self.bn(x + layer3)
            # final upsampling: here x16. This was fixed in the original paper

            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        elif self.n_upsample == 8: # use FCN-8s model

            # upsampling needs to be based on the num_classes feature maps

            # get intermediate layer and match channels
            if self.encoder_name in ["resnet18", "resnet34"]:
                layer3 = self.onebyone256(x["layer3"])
                layer2 = self.onebyone128(x["layer2"])
            else:
                layer3 = self.onebyone1024(x["layer3"])
                layer2 = self.onebyone512(x["layer2"])
            # upsample layer4 2x to match spatial resolution of layer3
            if self.b_bilinear:
                layer4 = F.interpolate(layer4, scale_factor=2.0, mode="bilinear", align_corners=False)
            else:
                layer4 = self.convTranspose(layer4)
            
            # concat both
            x = self.bn(layer3 + layer4)
            # upsample result of layer4 + 3 to match spatial resolution of layer2
            
            if self.b_bilinear:
                x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            else:
                x = self.convTranspose(x)
            x = self.bn(x + layer2)
            # final upsampling: here x8. This was fixed in the original paper
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        else:
            raise NotImplementedError(f"Upsampling of {self.n_upsample} is not implemented. Use either, 8, 16 or 32")
        return x



def load_fcn_resnet(encoder_name, num_classes=3, pretrained = False, replace_stride_with_dilation=False, n_upsample=32, b_bilinear=True):
    """
    Constructs a Fully-Convolutional Network model with a ResNet backbone.
    """

    if encoder_name in ["resnet18", "resnet34"]:
        head = FCNHead(512, num_classes)
    else:
        head = FCNHead(2048, num_classes)

    if replace_stride_with_dilation:
        backbone = load_resnet(
            encoder_name=encoder_name, 
            num_classes=num_classes, 
            pretrained=pretrained, 
            replace_stride_with_dilation=True
            )
        # set n_upsample =8, if we use dilated convolutions in the feature extractor --> no skip connections needed
        if n_upsample == 8:
            fcn = FCNtv(encoder_name, backbone, head, num_classes=num_classes, n_upsample=n_upsample, b_bilinear=b_bilinear, replace_stride_with_dilation=True)
        else:
            raise NotImplementedError(f"upsampling of {n_upsample} not implemented when using dilation instead of stride")
    else:
        backbone = load_resnet(
            encoder_name=encoder_name, 
            num_classes=num_classes, 
            pretrained=pretrained, 
            replace_stride_with_dilation=False
            )
        if n_upsample in [8, 16]:
            fcn = FCNskip(encoder_name, backbone, head, num_classes=num_classes, n_upsample=n_upsample, b_bilinear=b_bilinear, replace_stride_with_dilation=False)
        elif n_upsample == 32:
            fcn = FCNtv(encoder_name, backbone, head, num_classes=num_classes, n_upsample=n_upsample, b_bilinear=b_bilinear, replace_stride_with_dilation=False)
        else:
            raise NotImplementedError(f"upsampling of {n_upsample} not implemented when not using dilation")

    return fcn
