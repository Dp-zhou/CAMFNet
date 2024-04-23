import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse


from lib.CAMFNet import Network
from utils.dataloader import tt_dataset
from utils.eva_funcs import eval_Smeasure,eval_mae,numpy2tensor
import scipy.io as scio 
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--testsize',   type=int, default=352, help='the snapshot input size')
# parser.add_argument('--model_path', type=str, default='./checkpoints/')
parser.add_argument('--model_path', type=str, default='./Snapshot_b4_finallchose/CAMFNet/')
parser.add_argument('--save_path',  type=str, default='./results_145_finall/')

opt   = parser.parse_args()
model = Network().cuda()


cur_model_path = opt.model_path+'CAMFNet_145.pth'
model.load_state_dict(torch.load(cur_model_path))
model.eval()
        
    
################################################################

for dataset in ['CAMO', 'COD10K','CHAMELEON','NC4K']:#, 'CAMO', 'COD10K'
    
    save_path = opt.save_path + dataset + '/'
    os.makedirs(save_path, exist_ok=True)        
        
        
    test_loader = tt_dataset(r'/home/zbq/MSCAF-COD-master/TestDataset/{}/Imgs/'.format(dataset),
                               r'/home/zbq/MSCAF-COD-master/TestDataset/{}/GT/'.format(dataset), opt.testsize)
        

    with torch.no_grad():
        for iteration in range(test_loader.size):


            image, gt, name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()


            res,_,_,_,_ = model(image)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)


    ################################################################
            cv2.imwrite(save_path+name, res*255)
        


 
