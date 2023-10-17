import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, r = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(nn.Linear(in_channels, in_channels // r),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_channels // r, in_channels),
                                        nn.Sigmoid())

    def forward(self, x):
        SE = self.squeeze(x)
        SE = SE.reshape(x.shape[0],x.shape[1]) 
        SE = self.excitation(SE)
        SE = SE.unsqueeze(dim=2).unsqueeze(dim=3)
        x = x * SE
        return x
    

class SEBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1 ):
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * SEBottleneck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * SEBottleneck.expansion),
        )

        
        self.se_block = SEBlock(out_channels * SEBottleneck.expansion)

        self.projection_shortcut = nn.Sequential()

        if stride != 1 or in_channels != SEBottleneck.expansion * out_channels:
            self.projection_shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * SEBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * SEBottleneck.expansion),
            )
            
    def forward(self, x):
        residual= self.residual_function(x)
        se = self.se_block(residual)
        shortcut = self.projection_shortcut(x)
        out= nn.ReLU(inplace=True)(se + shortcut)   
        return out
    
class SEResNet(nn.Module):
    def __init__(self, block, num_block_list, num_classes):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2_x = self._make_layer(block, 64, num_block_list[0], stride=1)
        self.conv3_x = self._make_layer(block, 128, num_block_list[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, num_block_list[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, num_block_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x