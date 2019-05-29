import torch
from torch import nn
from torch.nn import functional as F
'''
编写函数时，务必要对齐了；otherwise error--Not imported forward
'''
# setting conv+bn+relu
class BasicConv2d(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size,stride=1,padding=0):
        super(BasicConv2d, self).__init__()
        #when left para,dont forget using '**kwargs'!!
        self.conv=nn.Conv2d(ch_in,ch_out,kernel_size,stride,padding)
        self.bn=nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x)

class Inception(nn.Module):
    def __init__(self,input_channel,n1,n3read,n3out,n5read,n5out,pool_block):
        super(Inception, self).__init__()
        #1*1conv branch
        self.b1=BasicConv2d(input_channel,n1,kernel_size=1)

        #1*1conv + 3*3conv
        self.b2_a=BasicConv2d(input_channel,n3read,kernel_size=1)
        self.b2_b=BasicConv2d(n3read,n3out,kernel_size=3,padding=1)

        #1*1conv + 3*3conv+ 3*3conv
        self.b3_a=BasicConv2d(input_channel,n5read,kernel_size=1)
        self.b3_b=BasicConv2d(n5read,n5out,kernel_size=3,padding=1)
        self.b3_c=BasicConv2d(n5out,n5out,kernel_size=3,padding=1)

        #pool

        self.b4_a=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.b4_b=BasicConv2d(input_channel,pool_block,kernel_size=1)

    def forward(self, x):
        x1=self.b1(x)
        x2=self.b2_b(self.b2_a(x))
        x3=self.b3_c(self.b3_b(self.b3_a(x)))
        x4=self.b4_b(self.b4_a(x))
        #lost return
        return torch.cat([x1,x2,x3,x4],dim=1)



class Googlenet(nn.Module):
    def __init__(self):
        super(Googlenet, self).__init__()

        self.pre_layers=BasicConv2d(3,192,kernel_size=3,padding=1)
        #pay attention to how to sure a correct variable name
        #for exam——self.3a is false name.
        self.a3=Inception(192,64,96,128,16,32,32)
        self.b3=Inception(256,128,128,192,32,96,64)

        self.max_pool=nn.MaxPool2d(3,stride=2,padding=1)

        self.a4=Inception(480,192,96,208,16,48,64)
        self.b4=Inception(512,160,112,224,24,64,64)
        self.c4=Inception(512,128,128,256,24,64,64)
        self.d4=Inception(512,112,144,288,32,64,64)

        self.e4=Inception(528,256,160,320,32,128,128)

        #self.max_pool2=nn.MaxPool2d(3,2,1)

        self.a5=Inception(832,256,160,320,32,128,128)
        self.b5=Inception(832,384,192,384,48,128,128)

        self.avg_pool=nn.AvgPool2d(8,stride=1)

        self.linear=nn.Linear(1024,10)

    def forward(self,x):
        out=self.pre_layers(x)
        #print(out.size())
        out=self.a3(out)
        out=self.b3(out)
        out=self.max_pool(out)
        out=self.a4(out)
        out=self.b4(out)
        out=self.c4(out)
        out=self.d4(out)
        out=self.e4(out)

        out=self.max_pool(out)
        out=self.a5(out)
        out=self.b5(out)
        out=self.avg_pool(out)

        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out

def main():
    #test Inception
    tmp=torch.randn(2,3,32,32)
    net1=Inception(3,3,3,6,3,3,3)
    out1=net1(tmp)
    print(out1.shape)
    #test Googlenet
    net2=Googlenet()
    out2=net2(tmp)
    print('out2 shape:',out2.shape)

if __name__=='__main__':
    main()
