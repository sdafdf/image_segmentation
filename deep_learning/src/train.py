#模型训练
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import os.path as osp
from model import ResNet18Unet
from transform import TrainTrainsform,TestTrainsform
from config import data_folder,mask_folder,batch_size,device,checkpoint,epoch_lr
from dataset import SegmentationData

def Train():
    net = ResNet18Unet().to(device=device)
    trainTrainsform = TrainTrainsform()
    trainset = SegmentationData(data_folder,mask_folder,trainTrainsform)
    testset = SegmentationData(data_folder,mask_folder,subset="test",trainsform=TestTrainsform())
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
    print(next(iter(trainloader)))
    testloader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=0)
    cirteron = nn.CrossEntropyLoss(weight=torch.Tensor([0.3,1.0]).to(device=device))
    best_loss = 1e9
    if osp.exists(checkpoint):
        ckpt = torch.load(checkpoint)
        best_loss = ckpt["loss"]
        net.load_state_dict(ckpt["params"])
        print("checkpoint loaded……")
    writer = SummaryWriter("logs")
    for n,(num_eopchs,lr) in enumerate(epoch_lr):
        optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=5e-3)
        for epoch in range(num_eopchs):
            net.train()
            pbar = tqdm(enumerate(trainloader),total=len(trainloader))
            epoch_loss = 0.0
            for i, (img,mask) in enumerate(trainloader):
                out = net(img.to(device))
                loss = cirteron(out,mask.to(device).long().squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i%10==0:
                    pbar.set_description("loss:{}".format(loss))
                epoch_loss+=loss.item()
                print("Epoch_loss:{}".format(epoch_loss/len(trainloader.dataset)))
                writer.add_scalar("seg_epoch_loss",epoch_loss/len(trainloader.dataset),sum([e[0] for e in epoch_lr[:n]])+epoch)
                with torch.no_grad():
                    net.eval()
                    test_loss = 0.0
                    for i,(img,mask) in tqdm(enumerate(testloader),total=len(testloader)):
                        out = net(img.to(device))
                        loss = cirteron(out,mask.to(device).long().squeeze(1))
                        test_loss+=loss.item()
                    print("Test_loss:{}".format(test_loss/len(testloader.dataset)))
                    writer.add_scalar(
                        "seg_test_loss",
                        test_loss/len(testloader.dataset),
                        sum([e[0] for e in epoch_lr[:n]])+epoch
                    )
                if test_loss<best_loss:
                    best_loss=test_loss
                    if not osp.isdir(checkpoint):
                        os.makedirs(checkpoint)
                    torch.save({"params":net.state_dict(),"loss":test_loss},checkpoint+"net.pth")
    writer.close()

if __name__=="__main__":
    Train()
