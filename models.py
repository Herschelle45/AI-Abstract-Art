import torch 
import torch.nn as nn 
class DCGANGenerator(nn.Module):
    def __init__(self,noise,channels,genfeatures):#genfeature = 64
        super().__init__()
        self.gen=nn.Sequential(
            nn.ConvTranspose2d(noise, genfeatures*64,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm2d(genfeatures*64),
            nn.ReLU(),
            nn.ConvTranspose2d(genfeatures*64, genfeatures*32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(genfeatures*32),
            nn.ReLU(),
            nn.ConvTranspose2d(genfeatures*32, genfeatures*16,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(genfeatures*16),
            nn.ReLU(),
            nn.ConvTranspose2d(genfeatures*16, genfeatures*8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(genfeatures*8),
            nn.ReLU(),
            nn.ConvTranspose2d(genfeatures*8, genfeatures*4,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(genfeatures*4),
            nn.ReLU(),
            nn.ConvTranspose2d(genfeatures*4, genfeatures*2,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(genfeatures*2),
            nn.ReLU(),
            nn.ConvTranspose2d(genfeatures*2,channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
    def forward(self,x):
        return self.gen(x)
class DCGANDiscriminator(nn.Module):
    def __init__(self,inchannels,disfeatures):#disfeature = 64
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(inchannels, disfeatures,kernel_size=4,stride=2,padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(disfeatures, disfeatures*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(disfeatures*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(disfeatures*2, disfeatures*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(disfeatures*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(disfeatures*4, disfeatures*8,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(disfeatures*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(disfeatures*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.disc(x)

def test(genclass, discclass):
    gen = genclass(100, 3, 64)
    disc = discclass(3, 64)
    geninput = torch.randn(32, 100, 1, 1)
    discinput = torch.randn(32, 3, 256, 256)
    print(gen(geninput).shape)
    print(disc(discinput).shape)
