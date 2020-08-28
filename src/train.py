#!/usr/bin/env python
'''
train foldingnet on nyc data

author  : Ruoyu Wang; Yuqiong Li
created : 10/25/18 1:29 PM
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from foldingnet import FoldingNetVanilla
from foldingnet import ChamfersDistance
from datasets import pcdDataset
from torch.utils.data import DataLoader
from utils import check_exist_or_remove, check_exist_or_mkdirs
from timeit import default_timer as timer
from datetime import timedelta
#torch.cuda.set_device(2)
#import os
#os.environ['CUDA_VISIBLE_DEVICES']='3'
#CUDA_VISIBLE_DEVICES = 1,2
#torch.cuda.set_device(2)


print("Entered before def train")

def train(dataset, model, batch_size, lr, epoches, log_interval, save_along_training):
    """train implicit version of foldingnet
    """
#    torch.cuda.set_device(3)
 #   CUDA_VISIBLE_DEVICES = 1,2
  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    chamfer_distance_loss = ChamfersDistance()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    print("Entered before model.train()")
    model = model.train()
 #   CUDA_VISIBLE_DEVICES = 2

    # enable distributed training
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    
    model.to(device)
    print("Entered after model.to(device)")
    check_exist_or_remove("/home/badri/bns332/foldingnet/log/train_loss_log.txt")
    check_exist_or_mkdirs("/home/badri/bns332/foldingnet/log")
    loss_log = open('../log/train_loss_log.txt', 'w+')
    running_loss=0.0
    start = timer()
    print("Entered after timer()")
    for ep in range(0, epoches):
        print("Epoch Number :",str(ep))
        print(running_loss)
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            opt.zero_grad()
            data = batch.to(device)
            # print(data.shape)
            #data = data.type(torch.cuda.FloatTensor)
            points_pred = model(data)
           # print("Entered after model(data)")
#            print("Shapes of input1 and input2 is ")
 #           print(data.size(),points_pred.size())
            loss = chamfer_distance_loss(data, points_pred)
            #print(loss.item())         
  #          print("Entered after computing chamfer loss")
            loss.backward()
            opt.step()
            #print(loss.item())
            running_loss += loss.item()
   #         print("Entered after running_loss+=loss.item()")
            if batch_idx % log_interval == log_interval - 1:
                end = timer()
                print('[%d, %5d] loss: %.6f elapsed time: %.2f' %
                    (ep + 1, batch_idx + 1, running_loss / log_interval, timedelta(seconds=end-start).total_seconds()))
                with open('../log/train_loss_log.txt', 'a+') as f:
                    f.write('[{0:d}, {1:5d}] loss: {2:.6f}\n'.format(ep + 1, batch_idx + 1, running_loss / log_interval))
                running_loss = 0.0
        if save_along_training:
            torch.save(model.state_dict(), os.path.join('/home/badri/bns332/foldingnet/model', 'ep_%d.pth' % ep))
    if save_along_training:   # the last one
        torch.save(model.state_dict(), os.path.join('/home/badri/bns332/foldingnet/model', 'ep_%d.pth' % ep))
    # loss_log.close()
    return


if __name__ == '__main__':
    # ROOT = "../data/nyc/"    # root path
    ROOT = "/data/RealCity3D/Finished_Pointclouds/New_York/Manhattan/Simple_Buildings"
    TRIAN_PATH = "/home/badri/bns332/foldingnet/train.txt"
    MLP_DIMS = (3,64,64,64,128,1024)
   # MLP_DIMS = (3,1024)
    FC_DIMS = (1024, 512, 512)
    #FC_DIMS = (1024,512)
    GRID_DIMS = (45, 45)
    
    FOLDING1_DIMS = (514, 512, 512, 3)
    #FOLDING1_DIMS = (514,3)
    FOLDING2_DIMS = (515, 512, 512, 3)
    #FOLDING2_DIMS = (515,512,3)
    MLP_DOLASTRELU = False

    check_exist_or_remove('/home/badri/bns332/foldingnet/model')   # clean up old history
    check_exist_or_mkdirs('/home/badri/bns332/foldingnet/model')
    kwargs = {
        'lr': 0.0001,
        # 330
        'epoches': 330,
        # 256
        'batch_size':256,
        'log_interval': 100,
        'save_along_training': True
    }

    with open(TRIAN_PATH) as fp:
        catelog = fp.readlines()
    catelog = [x.strip() for x in catelog]
    print("catelog done !")
    from open3d import *
    import datasets
    from datasets import plyDataset
    dataset = plyDataset(ROOT, catelog)
    print("About to create model")
    # model = FoldingNetShapes(MLP_DIMS, FC_DIMS, FOLDING1_DIMS, FOLDING2_DIMS)
    model = FoldingNetVanilla(MLP_DIMS, FC_DIMS, GRID_DIMS, FOLDING1_DIMS, FOLDING2_DIMS)
    print("About to start training")
    train(dataset, model, **kwargs)
    print("End training!!!")


