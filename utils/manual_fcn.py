
import torch
from torch import nn, Tensor
from manual_resnet import load_resnet
from torchinfo import summary

class FCNHead(nn.Sequential):
    def __init__(self, in_ch, out_ch, b_bn=False):
        """
        Caution: we still use Dropout, even tho not in the original implementation
        """
        mid_ch = in_ch // 4
        if b_bn:
            layers = [
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(mid_ch, out_ch, kernel_size=1),
            ]
        else:
            layers = [
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(mid_ch, out_ch, kernel_size=1),
            ]

        super().__init__(*layers)


class FCN(nn.Module):

    def __init__(self, encoder_name, backbone, head, num_classes=3, n_upsample=32, b_bilinear=True, replace_stride_with_dilation=False, b_bn=False):
        super().__init__()
        self.n_upsample = n_upsample
        self.encoder_name = encoder_name
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.b_bn = b_bn
        self.bn = nn.BatchNorm2d(num_features=num_classes)
        self.onebyone128 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.onebyone256 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.onebyone512 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.onebyone1024 = nn.Conv2d(1024, num_classes, kernel_size=1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=n_upsample)
        if b_bilinear:
            self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.upsample2 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, bias=False)
        
        self.backbone = backbone


        self.head = head

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        layer4 = self.head(x["layer4"])
        if self.n_upsample == 32: # use FCN-32s model
            print(f"{layer4.shape=}, {input_shape=}, using upsampling of {input_shape[-1]/layer4.shape[-1]}")
            x = self.upsample(layer4)
        elif self.n_upsample == 16: # use FCN-16s model
            # upsampling needs to be based on the num_classes feature maps

            # get intermediate layer and match channels
            if self.encoder_name in ["resnet18", "resnet34"]:
                layer3 = self.onebyone256(x["layer3"])
            else:
                layer3 = self.onebyone1024(x["layer3"])
            # upsample last layer 2x to match spatial resolution
            x = self.upsample2(layer4)
            # concat both
            if self.b_bn:
                print("using batch norm")
                x = self.bn(x + layer3)
            else:
                x = x+layer3
            x = self.upsample(x)

        elif self.n_upsample == 8: # use FCN-8s model
            
            if self.replace_stride_with_dilation: # if dilation is used, the output shape is x4 --> no skip connections needed
                x = self.upsample(layer4)
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
                layer3 = self.upsample2(layer3)
                layer4 = self.upsample2(layer4)
                layer4 = self.upsample2(layer4)

                # concat both
                if self.b_bn:
                    x = self.bn(layer2 + layer3)
                    x = self.bn(x + layer4)
                else:
                    x = layer2 + layer3
                    x = x + layer4

                # final upsampling: here x8
                x = self.upsample(x)


                # TODO: init deconv as bilinear

        else: 
            raise NotImplementedError(f"Upsampling of {self.upsample} is not implemented. Use either, 8, 16 or 32")


        return x



def load_fcn_resnet(encoder_name, num_classes=3, pretrained = False, replace_stride_with_dilation=False, n_upsample=32, b_bilinear=True, b_bn=True):
    """
    Constructs a Fully-Convolutional Network model with a ResNet backbone.
    """

    backbone = load_resnet(encoder_name=encoder_name, num_classes=num_classes, pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)
    if encoder_name in ["resnet18", "resnet34"]:
        head = FCNHead(512, num_classes)
    else:
        head = FCNHead(2048, num_classes)
    if replace_stride_with_dilation == True:
        n_upsample = 8 # TODO: need to do a case where we do x8 upsampling without skip connections
    return FCN(encoder_name, backbone, head, num_classes=num_classes, n_upsample=n_upsample, b_bilinear=b_bilinear, replace_stride_with_dilation=replace_stride_with_dilation, b_bn=b_bn)



def test():
    x = torch.randn(2, 3, 512, 512).to("cuda")
    for encoder_name in ["resnet18", "resnet34", "resnet50"]:
        model = load_fcn_resnet(encoder_name, 
        num_classes=3, 
        pretrained = True, 
        replace_stride_with_dilation=False, 
        n_upsample=16, 
        mode="bilinear",
        b_bn=False
        ).to("cuda")
        preds = model(x)


        summary(model, x.shape)
        assert x.shape == preds.shape, "shapes do not match"


if __name__ == "__main__":
    test()
