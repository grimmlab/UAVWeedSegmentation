from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from .manual_resnet import load_resnet
from .manual_unet import initialize_decoder, initialize_head
from segmentation_models_pytorch.encoders import get_encoder



class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class SeparableConv2d(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48   # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features


class DeepLabV3Head(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        layers = [
            nn.Dropout(0.1), # dropout was not used initially, but added to match FCN and reduce overfitting
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)]
        super().__init__(*layers)

class DLv3plus(nn.Module):
    """
    Reference:
        https://arxiv.org/abs/1802.02611v3

    """
    def __init__(
            self,
            encoder_name = "resnet34",
            pretrained = True,
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            num_classes: int = 3,
            upsampling: int = 4,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride)
            )
        #torch.Size([50, 3, 256, 256]), torch.Size([50, 64, 128, 128]), torch.Size([50, 64, 64, 64]), torch.Size([50, 128, 32, 32]), torch.Size([50, 256, 16, 16]), torch.Size([50, 512, 16, 16])
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights='imagenet',
            output_stride=encoder_output_stride,
        ) # --> stride 16. not 32

        self.backbone = load_resnet(
            encoder_name=encoder_name, 
            num_classes=num_classes, 
            pretrained=pretrained, 
            replace_stride_with_dilation=True
            )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = DeepLabV3Head(
            in_channels=self.decoder.out_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def forward(self, x):
        features = [self.backbone(x)[layer] for layer in ["layer0", "layer1", "layer2", "layer3", "layer4"]]
        #features = self.encoder(x)
        print([feature.shape for feature in features])
        
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks




def load_dl_resnet():
    pass

if __name__ == "__main__":
    load_dl_resnet()