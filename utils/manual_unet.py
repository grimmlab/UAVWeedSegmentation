# this script is inspired by the pytorch_semantic_segmentation package
import torch
import torch.nn as nn
import torch.nn.functional as F
from .manual_resnet import load_resnet

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= in_ch+skip_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels= out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(self, ch_per_block):
        super().__init__()

        blocks = [
            DecoderBlock(ch_per_block[0][0], ch_per_block[0][1], ch_per_block[0][2]),
            DecoderBlock(ch_per_block[1][0], ch_per_block[1][1], ch_per_block[1][2]),
            DecoderBlock(ch_per_block[2][0], ch_per_block[2][1], ch_per_block[2][2]),
            DecoderBlock(ch_per_block[3][0], ch_per_block[3][1], ch_per_block[3][2]),
            DecoderBlock(ch_per_block[4][0], ch_per_block[4][1], ch_per_block[4][2]),
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[::-1]  # start from top of feature extractor
        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

class UNetHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        layers = [
            nn.Dropout(0.1), # dropout was not used initially, but added to match FCN and reduce overfitting
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            ]
        super().__init__(*layers)

class UNet(nn.Module):
    def __init__(
        self,
        encoder_name = "resnet18",
        num_classes = 3,
        pretrained = True
    ):
        super().__init__()

        print(f"{encoder_name=}")
        if encoder_name in ["resnet18", "resnet34"]:
            # list of tuples for building different decoder blocks
            ch_per_block = [(512, 256, 256), (256, 128, 128), (128, 64, 64), (64, 64, 32), (32, 0, 16)]
        elif encoder_name in ["resnet50", "resnet101"]:
            ch_per_block = [(2048, 1024, 256), (256, 512, 128), (128, 256, 64), (64, 64, 32), (32, 0, 16)]
        
        self.backbone = load_resnet(
            encoder_name=encoder_name, 
            num_classes=num_classes, 
            pretrained=pretrained, 
            replace_stride_with_dilation=False
            )

        self.decoder = UNetDecoder(ch_per_block)

        self.segmentation_head = UNetHead(
            in_channels=ch_per_block[-1][-1],
            out_channels=num_classes,
            kernel_size=3,
        )

        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def forward(self, x):
        features = [self.backbone(x)[layer] for layer in ["layer0", "layer1", "layer2", "layer3", "layer4"]]
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks