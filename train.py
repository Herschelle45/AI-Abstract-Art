from models import DCGANDiscriminator, DCGANGenerator
import torch 
import torch.nn as nn 
import torchvision.transforms as t 
from dataset import AbstractArtDataset 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transforms = t.Compose([
        t.ToPILImage(),
        t.Resize((256,256)),
        t.ToTensor()
    ])
absartds = AbstractArtDataset(rootdir='../../abart', transforms=transforms)
absartloader = DataLoader(absartds, batch_size=8, shuffle=True) 

def weightinit(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module, 0.0, 0.02)
dcgangen = DCGANGenerator(100, 3, 64).to(device)
dcgandisc = DCGANDiscriminator(3, 64).to(device)
weightinit(dcgangen)
weightinit(dcgandisc)
optimgendcgan = torch.optim.Adam(dcgangen.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimdiscdcgan = torch.optim.Adam(dcgandisc.parameters(), lr=2e-4, betas=(0.5, 0.999))
EPOCHS = 15
lossfunc = torch.nn.BCELoss().to(device)
for epoch in range(EPOCHS):
  loop = tqdm(absartloader)
  for img in loop:
    img = img.to(device)
    noise = torch.randn(8, 100, 1, 1).to(device)
    fake = dcgangen(noise).to(device)
    discreal = dcgandisc(img).reshape(-1).to(device)
    discfake = dcgandisc(fake).reshape(-1).to(device)
    discrealloss = lossfunc(discreal, torch.ones_like(discreal))
    discfakeloss = lossfunc(discfake, torch.zeros_like(discfake))
    discloss = (discrealloss+discfakeloss)/2
    optimdiscdcgan.zero_grad()
    discloss.backward(retain_graph=True)
    optimdiscdcgan.step()
    pred = dcgandisc(fake).reshape(-1).to(device)
    genloss = lossfunc(pred, torch.ons_like(pred))
    optimgendcgan.zero_grad()
    genloss.backward()
    optimgendcgan.step()
  if (epoch+1)%5==0:
        torch.save({'model_state':dcgangen.state_dict()},f'absartgen_{epoch+1}.pt')
        torch.save({'model_state':dcgandisc.state_dict()},f'absartgen_{epoch+1}.pt')
