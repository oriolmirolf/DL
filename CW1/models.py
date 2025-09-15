import torch
import torch.nn as nn
import torch.nn.functional as F

class MaorNet(nn.Module):
    def __init__(self, num_classes=29, batch_norm=True, dropout=True, gaussian_initialization=False, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.bn1   = nn.BatchNorm2d(64)
        self.bn2   = nn.BatchNorm2d(64)

        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding="same")
        self.bn3   = nn.BatchNorm2d(128)
        self.bn4   = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.bn5   = nn.BatchNorm2d(256)
        self.bn6   = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding="same")
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding="same")
        self.bn7   = nn.BatchNorm2d(512)
        self.bn8   = nn.BatchNorm2d(512)

        
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(p=dropout_rate)
        
        self.fc1   = nn.Linear(512 * 14 * 14, 256)
        self.fc2   = nn.Linear(256, num_classes)

        if batch_norm == False:
            self.bn1 = self.bn2 = self.bn3 = self.bn4 = self.bn5 = self.bn6 = self.bn7 = self.bn8 = nn.Identity()
        if dropout == False:
            self.drop = nn.Identity()
    
        if gaussian_initialization:
            self.apply(self._init_weights)
        # else He initialization is used by default 

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # Third block
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool(x)

        # Fourth block
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True, dropout_rate=0.25):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

        self.drop  = nn.Dropout(p=dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DropPath(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob
        return x * mask / keep_prob

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, batch_norm=True, gaussian_initialization=False, p_head=0.5, p_block=0.5):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        self.head_drop = nn.Dropout(p=p_head)
        self.p_block = p_block
        
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        if gaussian_initialization:
            self.apply(self._init_weights)        

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.batch_norm, self.p_block))

            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        # Apply Gaussian initialization to Conv2d and Linear layers.
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.head_drop(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out