import os
os.remove('/home/badri/bns332/foldingnet/train.txt')
a = os.listdir('/data/RealCity3D/Finished_Pointclouds/New_York/Manhattan/Simple_Buildings')
fi = open('/home/badri/bns332/foldingnet/train.txt','a')

for files in a:
     fi.write(files+'\n')
     
#import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'
