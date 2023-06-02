import torch
import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    # self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=3,padding=1 )
        
    # 这里的in_channels和上面第一个卷积层的in_channels对应
    depth_conv0 = nn.Conv2d(in_channels=in_planes,out_channels=in_planes,kernel_size=3,stride=stride,padding=1,groups=in_planes)#groups对应out_channels
    # 这里的out_channels和上面第一个卷积层的out_channels对应
    point_conv0 = nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=1,padding=0)
    depthwise_separable_conv0 = torch.nn.Sequential(depth_conv0,point_conv0)
    # 上面3个替换了这一个卷积层



    # return nn.Conv2d(in_planes, out_planes, 
    #                  kernel_size = 3,stride = stride, 
    #                  padding = 1, bias = False)
    return depthwise_separable_conv0
    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion * 4, num_classes)
        self.fc = nn.Linear(512 * 1*1, num_classes)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
              nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False),
              nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def resnet18_dw_diy(num_class=26):
    model = ResNet(BasicBlock,[2,2,2,2],num_class)
    return model