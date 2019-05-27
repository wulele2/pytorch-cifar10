import torch
from torch import nn

class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()

        self.conv_unit=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),#后面有逗号
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,5,1,0),
            nn.AvgPool2d(2,2),
         )
        self.fc_unit=nn.Sequential(
            #此处这个2待确定。
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
        #这一部分用来测试经过卷积单元的输出的维度是多少。
        # tmp=torch.randn(2,3,32,32)
        # out=self.conv_unit(tmp)
        # print('conv_unit shape:',out.shape)

    def forward(self,x):
        #[b,3,32,32]
        batchsz=x.size(0)
        #[b,3,32,32]=>[b,16,5,5]
        x=self.conv_unit(x)
        #[b,16,5,5]=>[b,16*5*5]
        x=x.view(batchsz,-1)
        logits=self.fc_unit(x)
        return logits




def main():
    net=lenet()
    tmp=torch.randn(2,3,32,32)
    out=net(tmp)
    print('out size:',out.shape)

if __name__=='__main__':
    main()

