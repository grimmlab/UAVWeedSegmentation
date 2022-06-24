import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from .manual_resnet import load_resnet




class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn(x)
        return x



class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        encoder_channels = encoder_channels[::-1] # reversing, so we start with deepest channels
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, *features):
        x = features[0]
        skips = features[1:]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class UnetHead(nn.Sequential):

    def __init__(self, in_ch, out_ch, kernel_size=3):
        layers = [nn.Dropout(0.1),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
                ]
        super().__init__(*layers)


class Unet(nn.Module):
    def __init__(self, backbone, num_classes=3, encoder_channels=(64, 64, 128, 256, 512), decoder_channels= (256, 128, 64, 32, 16)):
        super().__init__()
        self.backbone = backbone

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )

        self.head = UnetHead(
            in_ch=decoder_channels[-1],
            out_ch=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        features = [self.backbone(x)[layer] for layer in ["layer4", "layer3", "layer2", "layer1","layer0"]]
        x = self.decoder(*features)
        x = self.head(x)
        return x



def load_unet_resnet(encoder_name, num_classes=3, pretrained = False):
    """
    Constructs a Fully-Convolutional Network model with a ResNet backbone.
    """
    decoder_channels= (256, 128, 64, 32, 16)
    if encoder_name in ["resnet18", "resnet34"]:
        encoder_channels=(64, 64, 128, 256, 512)
    elif encoder_name in ["resnet50", "resnet101"]:
        encoder_channels=(64, 256, 512, 1024, 2048)

    backbone = load_resnet(
            encoder_name=encoder_name, 
            num_classes=num_classes, 
            pretrained=pretrained, 
            replace_stride_with_dilation=False
            )

    unet = Unet(backbone, num_classes=num_classes, encoder_channels=encoder_channels, decoder_channels=decoder_channels)
    return unet 



def test():
    x = torch.randn(2, 3, 256, 256).to("cuda")
    for encoder_name in ["resnet18"]:# "resnet34", "resnet50", "resnet101"
        model = load_unet_resnet(
            encoder_name=encoder_name,
            num_classes=3,                      
        ).to("cuda")
        preds = model(x)
        summary(model, x.shape)
        print(f"{encoder_name}: {preds.shape=}")

if __name__ == "__main__":
    test()
