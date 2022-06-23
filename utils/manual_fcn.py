
import torch
from torch import nn
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

    def __init__(self, backbone, head, num_classes=3):
        super().__init__()
        self.encoder_name = "resnet50"
        self.bn = nn.BatchNorm2d(num_features=num_classes)
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)["layer4"]
        x = self.head(x)
        x = self.bn(x)
        # other function used here in torchvision implementation
        x = nn.functional.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x


class FCN(nn.Module):

    def __init__(self, encoder_name, backbone, head, num_classes=3, n_upsample=32, b_bilinear=True, replace_stride_with_dilation=False, b_bn=False):
        super().__init__()
        self.n_upsample = n_upsample
        self.encoder_name = encoder_name
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.b_bilinear=b_bilinear
        self.b_bn = b_bn
        self.bn = nn.BatchNorm2d(num_features=num_classes)
        self.onebyone128 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.onebyone256 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.onebyone512 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.onebyone1024 = nn.Conv2d(1024, num_classes, kernel_size=1)
        if not b_bilinear:
            self.convTranspose = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, bias=False)
        
        self.backbone = backbone


        self.head = head

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        layer4 = self.head(x["layer4"])
        if self.n_upsample == 32: # use FCN-32s model
            #if self.b_bilinear:
            x = nn.functional.interpolate(layer4, scale_factor=self.n_upsample, mode='bilinear', align_corners=None)
            #else:
            #    x = self.convTranspose(layer4)
        
        elif self.n_upsample == 16: # use FCN-16s model
            # upsampling needs to be based on the num_classes feature maps

            # get intermediate layer and match channels
            if self.encoder_name in ["resnet18", "resnet34"]:
                layer3 = self.onebyone256(x["layer3"])
            else:
                layer3 = self.onebyone1024(x["layer3"])
            # upsample last layer 2x to match spatial resolution
            if self.b_bilinear:
                x = nn.functional.interpolate(layer4, scale_factor=2, mode='bilinear', align_corners=None)
            else:
                x = self.convTranspose(layer4)
            # concat both
            if self.b_bn:
                print("using batch norm")
                x = self.bn(x + layer3)
            else:
                x = x+layer3
                
            x = nn.functional.interpolate(x, scale_factor=self.n_upsample, mode='bilinear', align_corners=None)

        elif self.n_upsample == 8: # use FCN-8s model
            
            if self.replace_stride_with_dilation: # if dilation is used, the output shape is x4 --> no skip connections needed
                x = nn.functional.interpolate(layer4, scale_factor=self.n_upsample, mode='bilinear', align_corners=None)
            else:
                # upsampling needs to be based on the num_classes feature maps

                # get intermediate layer and match channels
                if self.encoder_name in ["resnet18", "resnet34"]:
                    layer3 = self.onebyone256(x["layer3"])
                    layer2 = self.onebyone128(x["layer2"])
                else:
                    layer3 = self.onebyone1024(x["layer3"])
                    layer2 = self.onebyone512(x["layer2"])
                # upsample last layer 2x to match spatial resolution
                #layer3 = self.upsample2(layer3) # here bug in train code but not in test down here. changed to interpolate, as upsample is depecated
                layer3 = nn.functional.interpolate(layer3, scale_factor=2, mode='bilinear', align_corners=None)
                layer4 = nn.functional.interpolate(layer4, scale_factor=4, mode='bilinear', align_corners=None)

                # concat both
                if self.b_bn:
                    x = self.bn(layer2 + layer3)
                    x = self.bn(x + layer4)
                else:
                    x = layer2 + layer3
                    x = x + layer4

                # final upsampling: here x8
                x = nn.functional.interpolate(x, scale_factor=self.n_upsample, mode='bilinear', align_corners=None)


                # TODO: init deconv as bilinear

        else: 
            raise NotImplementedError(f"Upsampling of {self.n_upsample} is not implemented. Use either, 8, 16 or 32")


        return x



def load_fcn_resnet(encoder_name, num_classes=3, pretrained = False, replace_stride_with_dilation=False, n_upsample=32, b_bilinear=True):
    """
    Constructs a Fully-Convolutional Network model with a ResNet backbone.
    """

    backbone = load_resnet(encoder_name=encoder_name, num_classes=num_classes, pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)
    
    
    if encoder_name in ["resnet18", "resnet34"]:
        head = FCNHead(512, num_classes)
    else:
        head = FCNHead(2048, num_classes)
        fcn = FCNtv(backbone, head)
        # TODO: make it work with other models than resnet50 
    #if replace_stride_with_dilation == True:

    #    print("replacing stride")
    #    n_upsample = 8
    #fcn = FCN(encoder_name, backbone, head, num_classes=num_classes, n_upsample=n_upsample, b_bilinear=b_bilinear, replace_stride_with_dilation=replace_stride_with_dilation, b_bn=b_bn)
    return fcn 



def test():
    x = torch.randn(2, 3, 256, 256).to("cuda")
    for encoder_name in ["resnet50"]:
        for n_upsample in [8]:
            for replace_stride in [True]:
                model = load_fcn_resnet(encoder_name, 
                num_classes=3, 
                pretrained = True, 
                replace_stride_with_dilation=replace_stride, 
                n_upsample=n_upsample, 
                b_bilinear=True,
                b_bn=False
                ).to("cuda")
                preds = model(x)
                print(f"{encoder_name} n_upsample: {n_upsample} replace: {replace_stride}:{preds.shape=}")


                #assert x.shape == preds.shape, "shapes do not match"


if __name__ == "__main__":
    test()
