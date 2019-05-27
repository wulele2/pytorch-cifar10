import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn,optim
#命名为lenet-5时报错
from lenet5 import lenet

def main():
    batchsz=32
    #load train
    cifar_train=datasets.CIFAR10('cifar',train=True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        ]),download=True)
    cifar_train=DataLoader(cifar_train,batch_size=batchsz,shuffle=True)
    #load test
    cifar_test=datasets.CIFAR10('cifar',train=False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        ]),download=True)
    cifar_test=DataLoader(cifar_train,batch_size=batchsz,shuffle=False)

    x,label=iter(cifar_train).next()
    print('x:',x.shape,'label:',label.shape)

    model=lenet()

    creterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    print(model)
    #在循环中分为两个部分。一部分model.train;model.eval
    for epoch in range(100):
        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):
            logits=model(x)
            loss=creterion(logits,label)

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch,'loss:',loss)

        model.eval()
        with torch.no_grad():
            total_num=0
            total_correct=0
            for x,label in cifar_test:
                logits=model(x)
                #[b,10]
                pred=torch.argmax(dim=1)
                correct=torch.eq(pred,label).float().sum().item()
                total_correct+=correct
                total_num+=x.size(0)
                acc=total_correct/total_num

            print(epoch,'test acc',acc)


if __name__=='__main__':
    main()

