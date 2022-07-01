from models import Discriminator, Generator
import torch 
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
absartds = AbstractArtDataset(rootdir='/Users/herschelle/coding/Python/Data_Science/abart', transforms=transforms)
absartloader = DataLoader(absartds, batch_size=8, shuffle=True) 
gen = Generator(100, 3, 64)
disc = Discriminator(3, 64)
optimgen = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999)) 
optimdisc = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999)) 
EPOCHS=5
lossfunc = torch.nn.BCELoss().to(device)
for epoch in range(EPOCHS):
  loop = tqdm(absartloader)
  for img in loop:
    img = img.to(device)
    noise = torch.randn(8, 100, 1, 1).to(device)
    fake = gen(noise).to(device)
    discreal = disc(img).reshape(-1).to(device)
    discfake = disc(fake).reshape(-1).to(device)
    discrealloss = lossfunc(discreal, torch.ones_like(discreal))
    discfakeloss = lossfunc(discfake, torch.zeros_like(discfake))
    discloss = (discrealloss+discfakeloss)/2
    optimdisc.zero_grad()
    discloss.backward(retain_graph=True)
    optimdisc.step()
    pred = disc(fake).reshape(-1).to(device)
    genloss = lossfunc(pred, torch.ones_like(pred))
    optimgen.zero_grad()
    genloss.backward()
    optimgen.step()
