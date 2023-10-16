import torch 
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False),
        )
        
    def forward(self, x):
        return torch.cat([x, self.residual(x)], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.transition = nn.Sequential(nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                        nn.AvgPool2d(2))

    def forward(self, x):
        return self.transition(x)
    
    
class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, growth_rate=12, reduction=0.5):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate
        
        self.stem= nn.Sequential(nn.Conv2d(3, inner_channels, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(inner_channels),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        layers= []
        
        for num_block in num_blocks[:-1]:
            layers += [self._make_layers(inner_channels, num_block)]
            inner_channels += self.growth_rate * num_block
            out_channels = int(reduction * inner_channels)
            layers += [Transition(inner_channels, out_channels)]
            inner_channels = out_channels
        
        layers+= [self._make_layers(inner_channels, num_blocks[-1])]
        inner_channels += self.growth_rate * num_blocks[-1]
        layers += [nn.BatchNorm2d(inner_channels),]
        layers += [nn.ReLU(inplace=True),]
        
        self.features= nn.Sequential(*layers)
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.classifier= nn.Linear(inner_channels, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
    
    def _make_layers(self, in_channels, num_block):
        layers= []
        for i in range(num_block):
            layers.append(Bottleneck(in_channels, self.growth_rate))
            in_channels += self.growth_rate
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x= self.stem(x)
        x= self.features(x)
        x= self.avgpool(x)
        x= x.view(x.size(0), -1)
        x= self.classifier(x)
        
        return x