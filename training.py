from modules.leaky import LIF
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
u_th, beta = 1.0, 0.5
timestamp = 10
# from torch.utils.tensorboard import SummaryWriter

class LIFSNN(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.pool1 = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(32)
        self.snn1 = LIF(beta=beta, u_th=u_th)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool2 = nn.MaxPool2d(2)
        self.norm2 = nn.BatchNorm2d(64)
        self.snn2 = LIF(beta=beta, u_th=u_th)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1600, 10)
        self.snn3 = LIF(beta=beta, u_th=u_th)
        
    def forward(self, inputs):
        LIF.reset()
        output = []
        for input in inputs:
            x = input
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.norm1(x)
            x = self.snn1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.norm2(x)
            x = self.snn2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.snn3(x)
            output.append(x)
        
        return torch.stack(output)
    
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

if __name__ == '__main__':
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True,transform=transforms)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transforms)
    
    mnist_train_iter = DataLoader(dataset=mnist_train, batch_size=64, shuffle=True, num_workers=4)
    mnist_test_iter = DataLoader(dataset=mnist_test, batch_size=64, shuffle=False, num_workers=4)
    
    net = LIFSNN().cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=2e-3)
    
    for epoch in range(3):
        i = 0
        for input, labels in mnist_train_iter:
            inputs = torch.stack([input]*timestamp)
            optimizer.zero_grad()
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = net(inputs)
            prob = torch.mean(outputs, dim=0)
            loss = torch.nn.functional.cross_entropy(prob, labels)
            acc = (torch.argmax(prob, dim=-1) == labels).float().mean()
            loss.backward()
            optimizer.step()
            i+=1
            if i % 100 ==0:
                print(f'epoch:{epoch} tran_loss: {loss.item()}, acc: {acc.item()}')
            
        with torch.no_grad():  
            corr = 0  
            total = 0
            for input, labels in mnist_test_iter:
                inputs = torch.stack([input]*timestamp)
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                outputs = net(inputs)
                prob = torch.mean(outputs, dim=0)
                total += labels.size(0)
                corr += (torch.argmax(prob, dim=-1) == labels).sum().item()
            print(f'test acc:', corr*1.0/total*100)
            
            torch.save(net.state_dict(),'./model_files/mnist_snn_{}.pth'.format(epoch))