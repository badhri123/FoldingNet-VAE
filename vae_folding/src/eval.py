#!/usr/bin/env python
'''
evaluate point cloud generative results

author  : Yuqiong Li, Ruoyu Wang
created : 05/02/19  10:30AM
'''
import sys
import os
import torch
import torch.nn as nn
from datasets import plyDataset
from foldingnet import FoldingNetVanilla, FoldingNetShapes
from torch.utils.data import DataLoader
import torch
from foldingnet import ChamfersDistance
from datasets import pcdDataset
import numpy as np
from utils import check_exist_or_remove
import scipy


def eval(dataset, model, batch_size, log_interval):
    """test implicit version of foldingnet
    TODO: return indices of a specific sample for comparison with meshes and voxels too
    """
#    batch_size=256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    chamfer_distance_loss = ChamfersDistance()
    model = model.eval().cuda()
    check_exist_or_remove("/home/badri/bns332/vae_folding/log/eval_loss_log.txt")
    loss_log = open('/home/badri/bns332/vae_folding/log/eval_loss_log.txt', 'w')
    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        data = batch.cuda()
        print("Batch size",data.size())
        codes = torch.load('codes.pt')
        points_pred,mu,std = model(data,codes)
#        break
#    return points_pred,mu,std,batch
      #  np.save("../val/o{}.npy".format(batch_idx), data.cpu().detach().numpy())   # original
       # np.save("../val/p{}.npy".format(batch_idx), points_pred.cpu().detach().numpy())    # predicted
        print("Came Here")
        loss = chamfer_distance_loss(data, points_pred)
        running_loss += loss.item()
#        if batch_idx % log_interval == log_interval - 1:
 #           print('[%d, %5d] loss: %.6f' %
  #          (1, batch_idx + 1, running_loss / log_interval))
   #         print('[%d, %5d] loss: %.6f' %
    #        (1, batch_idx + 1, running_loss / log_interval), file=loss_log)
     #       running_loss = 0.0
        break

    #loss_log.close()
    return points_pred,mu,std,batch



def main(modelpath):
    ROOT = "/data/RealCity3D/Finished_Pointclouds/New_York/Manhattan/Simple_Buildings"    # root path
    TEST_PATH = "/home/badri/bns332/vae_folding/train.txt"    # test path
    MLP_DIMS = (3,64,64,64,128,1024)
    FC_DIMS = (1024, 512, 512)
    GRID_DIMS = (64, 64)
    FOLDING1_DIMS = (514, 512, 512, 3)   # change the input feature of the first fc because now has 9 dims instead of 2
    FOLDING2_DIMS = (515, 512, 512, 3)
    MLP_DOLASTRELU = False
   # codes = torch.load('codes.pt')

    with open(TEST_PATH) as fp:
        catelog = fp.readlines()
    catelog = [x.strip() for x in catelog]
    print("catelog done !")

    testset = plyDataset(ROOT, catelog)
    model = FoldingNetVanilla(MLP_DIMS, FC_DIMS, GRID_DIMS, FOLDING1_DIMS, FOLDING2_DIMS)   
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(modelpath))
    batch_size = 256
    log_interval = 10
    points_pred,mu,std,batch = eval(testset, model, batch_size, log_interval)
    print("End evaluation!!!")
    return points_pred,mu,std,batch

#if __name__ == "__main__":
#print(*sys.argv[1:])    
points_pred,mu,std,batch = main('/home/badri/bns332/vae_folding/model/ep_104.pt')

print(points_pred.size(),mu.size(),std.size(),batch.size())
import numpy as np
import open3d
from open3d import *
#torch.save(points_pred, 'reconstructedpoints.pt')
#torch.save(batch,'original.pt')
#points = points_pred[0].cpu().data.numpy()
#points = points.reshape((2025,3))
#DISPLAY = 0
#point_cloud = open3d.geometry.PointCloud()
#point_cloud.points = open3d.utility.Vector3dVector(points)
#pcd = open3d.io.read_point_cloud("/data/RealCity3D/Finished_Pointclouds/New_York/Manhattan/Simple_Buildings/gml_I6ETIGTEU6GKWNAT2Y2J9H6KLI34WHY1F0V8.ply")
#open3d.visualization.draw_geometries([point_cloud])
#import viz
#from viz import vis_points
#vis_points(point_cloud)
