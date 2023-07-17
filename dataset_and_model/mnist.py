from torch import nn
import torch
import snntorch as snn
from snntorch import surrogate, functional, utils
import torchvision
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
from modules.gumbel_softmax import GumbelSampler
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class NMNIST:
    def __init__(self) -> None:
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

        

class NMNIST_MODEL(nn.Module):
    def __init__(self) -> None:
        spike_grad = surrogate.atan()
        beta = 0.5
        
        self.net = nn.Sequential(
                            nn.Conv2d(2, 32, 3,1,1),
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
    
def get_dataset_and_model(dataset_path:str, model_path:str, num_pic:int):
    data = NMNIST()
    model = NMNIST_MODEL()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print()
    
    return data, model 
    
def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)
    
    for step in range(data.size(0)):
        data_gumbel_softmax = GumbelSampler(data[step])
        spk_out, mem_out = net(data_gumbel_softmax)
        spk_rec.append(spk_out)
        
    return torch.stack(spk_rec)