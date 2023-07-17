import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import snntorch as snn
from snntorch import surrogate, functional, utils
from torch import nn
import torch
import torchvision
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset

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
    event, targets = next(iter(trainloader))
    print(event.shape)
    event, targets = next(iter(testloader))
    print(event.shape)

    return trainloader, testloader

def load_model():
    
    spike_grad = surrogate.atan()
    beta = 0.5
    
    net = nn.Sequential(nn.Conv2d(2, 32, 3,1,1),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(2), 
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        
                        nn.Conv2d(32, 64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2), 
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        
                        nn.Conv2d(64,128,3,1,1),
                        nn.AdaptiveAvgPool2d((2,2)),
                        nn.Flatten(),
                        nn.Linear(128*2*2, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)
    return net

def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)
    
    for step in range(data.size(0)):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        
    return torch.stack(spk_rec)



def train(train_loader, test_loader, net):
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))
    loss_fn = functional.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)    
    acc_fn = functional.accuracy_rate
    num_epoch=10
    loss_hist=[]
    acc_hist=[]
    
    for epoch in range(num_epoch):
        for i , (data, targets) in enumerate(iter(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            
            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            loss_hist.append(loss_val.item())
            
            if i % 50 == 0:
                print(f'epoch:{epoch}, iteration:{i}, train loss:{loss_val.item():.2f}')
                acc = acc_fn(spk_rec, targets)
                acc_hist.append(acc)
                print(f'train_accuracy:{acc*100:.2f}%')
            
        net.eval()
        with torch.no_grad():
            total_acc = 0
            for i , (data, targets) in enumerate(iter(test_loader)):
                data = data.to(device)
                targets = targets.to(device)
                
                output = forward_pass(net, data)
                acc = acc_fn(output, targets)
                total_acc += acc.item()
                
            total_acc /= len(test_loader)
            print(f'test acc:{total_acc*100:.2f}\n')
            torch.save(net.state_dict(),f'./model_files/Pth{epoch}.pth')
    
    
if __name__ == '__main__':
    train_loader, test_loader = load_data()
    net = load_model()
    train(train_loader, test_loader, net)