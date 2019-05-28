import torch
from torch import nn
from torch.nn import functional as F

class Resblk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(Resblk, self).__init__()

        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)
        #以下部分主要解决 输入通道和输出通道 不一致的情况
        self.extra=nn.Sequential()
        if ch_in != ch_out:
            #将ch_in 的维度 转换成 ch_out
            self.extra=nn.Sequential(nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                                     nn.BatchNorm2d(ch_out)
                                     )
    #pay attention to this fraction,espeicially 'out' and 'x' ' channel is different.
    #oterwise lead to channel is not diffenrent bug.
    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))

        out=out+self.extra(x)
        out=F.relu(out)
        return out

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        #input [b,3,32,32]=>[b,16,x,x]
        self.conv1=nn.Sequential(nn.Conv2d(3,16,kernel_size=3,stride=3,padding=0),
                                 nn.BatchNorm2d(16)
                                 )
        self.blk1=Resblk(16,32,stride=2)
        self.blk2=Resblk(32,64,stride=2)
        self.blk3=Resblk(64,128,stride=2)
        self.blk4=Resblk(128,256,stride=2)

        self.outlayer=nn.Linear(800*1*1,10)

    def forward(self, x):
        #[b,3]=>[b,16]
        x=F.relu(self.conv1(x))

        x=self.blk1(x)
        x=self.blk2(x)
        x=self.blk3(x)
        x=self.blk4(x)

        #
        x=x.view(x.size(0),-1)
        #pytorch can random debug and print the info of x
        #print(x.shape)

        #when u use outlayer,must keep the dim same，
        #x.size(0)==512,so the self.outlayer's first dim keep the same.
        out=self.outlayer(x)

        return out
def main():
    #test Resblk
    # tmp1=torch.randn(2,16,32,32)
    # blk=Resblk(16,32)
    # out1=blk(tmp1)
    # print('out1 shape:',out1.shape)
    
    #test resnet18
    tmp=torch.randn(2,3,32,32)
    net=resnet18()
    out=net(tmp)
    print('out.shape:',out.shape)

if __name__=='__main__':
    main()
