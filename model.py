import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    """  (Squeeze-and-Excitation block) """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """ Spatia Attention Module"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        
        x = self.conv1(x)
        
        return x * self.sigmoid(x)

class ResNet18WithAttention(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet18WithAttention, self).__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
        self.channel_attention = ChannelAttention(self.base_model.layer4[-1].conv2.out_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.channel_attention(x)
        attention_map = self.spatial_attention(x)
        x = x*attention_map.expand_as(x)
        x = self.base_model.avgpool(x)
        
        
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x
    

    
    
class ResNet34WithAttention(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet34WithAttention, self).__init__()
        
        self.base_model = models.resnet34(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
        self.channel_attention = ChannelAttention(self.base_model.layer4[-1].conv2.out_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.channel_attention(x)
        attention_map = self.spatial_attention(x)
        x = x*attention_map.expand_as(x)

        
        

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x
    

class ResNet50WithAttention(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet50WithAttention, self).__init__()
        
        self.base_model = models.resnet50(pretrained=True)
        
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
        
        self.channel_attention = ChannelAttention(self.base_model.layer4[-1].conv3.out_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

       
        x = self.channel_attention(x)
        attention_map = self.spatial_attention(x)
        x = x*attention_map.expand_as(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x