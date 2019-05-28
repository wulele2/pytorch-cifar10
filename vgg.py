import torch
from torch import nn

class vggnet(nn.Module):
    def __init__(self):
        super(vggnet, self).__init__()

        self.unit1=nn.Sequential(
            nn.Conv2d(3,3,3,stride=1,padding=1),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(2,2),
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
        )
        self.unit2=nn.Sequential(
            nn.Linear(256*4*4,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    def forward(self, x):
        #[2,3,32,32]=>[2,256,4,4]
        x=self.unit1(x)
        x=x.view(x.size(0),-1)
        x=self.unit2(x)

        return x

def main():
    tmp=torch.randn(2,3,32,32)
    net=vggnet()
    out=net(tmp)
    print('out shape:',out.shape)

if __name__=='__main__':
    main()