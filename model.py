import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt



def visulization(images, labels, result_path='test.png'):
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    print(labels)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(result_path, pad_inches = 0, bbox_inches = 'tight')
    plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128, 64, bias=False),
            nn.Sigmoid(),
            nn.Linear(64, 16, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64, bias=False),
            nn.Sigmoid(),
            nn.Linear(64, 128, bias=False),
            nn.Sigmoid(),
            nn.Linear(128, 28*28, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DCW_woConstraint(nn.Module):
    def __init__(self):
        super(DCW_woConstraint, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 64, bias=False)
        self.l3 = nn.Linear(64, 16, bias=False)
        self.wlist = [self.l1.weight, self.l2.weight, self.l3.weight]
    def forward(self, x):
        w3 = self.l3.weight.T
        w2 = self.l2.weight.T
        w1 = self.l1.weight.T

        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = self.l3(x)
        x = F.leaky_relu(F.linear(x, w3))
        x = F.leaky_relu(F.linear(x, w2))
        x = F.linear(x, w1)
        return x

class DCW(nn.Module):
    def __init__(self):
        super(DCW, self).__init__()
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 64, bias=False)
        self.l3 = nn.Linear(64, 16, bias=False)
        self.wlist = [self.l1.weight, self.l2.weight, self.l3.weight]
        self.dew = [None]*3
    def forward(self, x):
        for i in range(3):
            (u, s, v) = torch.svd(self.wlist[i])
            self.dew[i]=v.mm(torch.eye(min(u.size()))*(1/s.unsqueeze(1))).mm(u.T)
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = self.l3(x)
        x = F.leaky_relu(F.linear(x, self.dew[2]))
        x = F.leaky_relu(F.linear(x, self.dew[1]))
        x = F.linear(x, self.dew[0])
        return x

if __name__ == "__main__":
    epochs = 50
    batch_size = 64
    lr = 0.003
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = DCW_woConstraint()
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()

    # training
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784)
            out = model(images)

            optim.zero_grad()
            loss = criterion(out, images)
            loss.backward()
            optim.step()
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())

    # testing
    images, labels = next(iter(test_loader))
    images = images.view(-1, 784)
    model.eval()
    out = model(images).view(batch_size, 1, 28, 28)
    visulization(out.detach(), labels)