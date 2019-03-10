import torch
import json
import os
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from model import Model
import torch.nn.utils.rnn as rnn
import image_provider
from tqdm import tqdm
import cv2
from skimage.io import imsave
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args
def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')

    #dataset_train = DataProvider(split='train', opts=opts['train'])
    dataset_test = DataProvider(split='val', opts=opts['train_val'],mode="train_val")

    #train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],
     #   shuffle = True, num_workers=opts['train']['num_workers'], collate_fn=manuscript.collate_fn)

    test_loader = DataLoader(dataset_test, batch_size=opts['train_val']['batch_size'],
        shuffle = False, num_workers=opts['train_val']['num_workers'], collate_fn=image_provider.collate_fn)
    
    return test_loader

class Tester(object):
    def __init__(self, args,opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts
        self.test_loader = get_data_loaders(self.opts['dataset'],image_provider.DataProvider)
        self.model = Model(64,64*12,3).to(device)  
        self.model.load_state_dict(torch.load(self.opts["model_path"])["state_dict"])
    def test(self):
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.test_loader)):
                img = data['img']
          
                img = torch.cat(img)
                img = img.view(-1,64,768,3)
                img = torch.transpose(img,1,3)
                img = torch.transpose(img,2,3)
                edge_mask = data["edge_mask"]
                fp_mask = data["fp_mask"]
                #fp_mask = fp_mask.view(fp_mask.shape[0],-1)
                img = img.float()
                
                edge_logits,fp_logits= self.model(img.cuda())
                
                fp_logits = fp_logits.view(fp_mask.shape)
                fp_logits = fp_logits.cpu().numpy()
                #print(fp_logits)
                for pred_edge_mask in  edge_logits:
                    for i in range(len(pred_edge_mask)):
                        for j in range(len(pred_edge_mask[i])):
                            if pred_edge_mask[i][j]>0.5:
                                pred_edge_mask[i][j]=1
                            else:
                                pred_edge_mask[i][j]=0
                #print(fp_logits)
                #print(fp_mask)
               
                for pred_fp_logits in fp_logits:
                    max=0.0
                    index_i=0
                    index_j=0
                    for i in range(len(pred_fp_logits)):
                        for j in range(len(pred_fp_logits[i])):
                            if pred_fp_logits[i][j]>=max:
                                max = pred_fp_logits[i][j]
                                index_i=i
                                index_j=j
                            pred_fp_logits[i][j]=0.0
                    pred_fp_logits[index_i][index_j]=1.0
                print(pred_fp_logits)
                imsave("/home/jatin/image_classifier/data/penimages/"+str(step)+".jpg",pred_fp_logits)
                imsave("/home/jatin/image_classifier/data/penimages/"+str(step)+"gt.jpg",fp_mask[0])
if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    opts["model_path"] = '/home/jatin/image_classifier/checkpoints_maxpoolwithoutdropout/epoch14_step7000.pth' 
    tester = Tester(args,opts)
    tester.test()

