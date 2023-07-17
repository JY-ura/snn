import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate, functional, utils
import torchvision
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
from modules.gumbel_softmax import ArgSoftmax

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection if the input and output dimensions don't match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(16, 3, stride=1)
        self.layer2 = self.make_layer(32, 3, stride=2)
        self.layer3 = self.make_layer(64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def load_data():
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size,
                        time_window=10000)
    ])

    transform = tonic.transforms.Compose([
        torch.from_numpy,
        torchvision.transforms.RandomRotation([-10,10])
    ])

    train_dataset = tonic.datasets.NMNIST(save_to='./data',train=True, transform=frame_transform)
    test_dataset = tonic.datasets.NMNIST(save_to='./data',train=False, transform=frame_transform)

    cached_trainset = DiskCachedDataset(train_dataset, transform=transform, cache_path='./cache/nmnist/train')
    cached_testset = DiskCachedDataset(test_dataset, cache_path='./cache/nmnist/test')

    batch_size=128
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn= tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn= tonic.collation.PadTensors(batch_first=False))
    # event, targets = next(iter(testloader))
    # print(event.shape)

    return trainloader, testloader

def train(train_loader, test_loader, net):
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))
    loss_fn = functional.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)    
    acc_fn = functional.accuracy_rate
    num_epoch=1
    
    for epoch in range(num_epoch):
        net.train()
        for i , (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            

            spk_rec = net(data)
            loss_val = loss_fn(spk_rec, targets)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # loss_hist.append(loss_val.item())
            
            if i % 50 == 0:
                print(f'epoch:{epoch}, iteration:{i}, train loss:{loss_val.item():.2f}')
                acc = acc_fn(spk_rec, targets)
                # acc_hist.append(acc)
                print(f'accuracy:{acc*100:.2f}%')
            
        net.eval()
        with torch.no_grad():
            total_acc = 0
            for i , (data, targets) in enumerate(iter(test_loader)):
                data = data.to(device)
                targets = targets.to(device)
                
                output = net(data)
                acc = acc_fn(output, targets)
                total_acc += acc.item()
                
            total_acc /= len(test_loader)
            print(f'epoch:{epoch}, test loss:{total_acc}')
            # torch.save(net.state_dict(),'./model_file/Pth.pth')

if __name__ == '__main__':
    train_loader, test_loader = load_data()
    net = ResNet()
    train(train_loader, test_loader, net)