import torch
import json
import os
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from model import Model
import image_provider
from tqdm import tqdm
import torch.nn.functional as F
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("!!!!!!got cuda!!!!!!!")
else:
    print("!!!!!!!!!!!!no cuda!!!!!!!!!!!!")


def create_folder(path):
   # if os.path.exists(path):
       # resp = input 'Path %s exists. Continue? [y/n]'%path
    #    if resp == 'n' or resp == 'N':
     #       raise RuntimeError()
    
   # else:
     os.system('mkdir -p %s'%(path))
     print('Experiment folder created at: %s'%(path))
        
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print('Building dataloaders')

    dataset_train = DataProvider(split='train', opts=opts['train'],mode='train')
    dataset_val = DataProvider(split='train_val', opts=opts['train_val'],mode='train_val')
    #weight = dataset_train.getweight()
    #label_to_count = dataset_train.getlabeltocount()
    #label_to_countval = dataset_val.getlabeltocount()
    train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],
        shuffle=True, num_workers=opts['train']['num_workers'], collate_fn=image_provider.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
        shuffle = False, num_workers=opts['train_val']['num_workers'], collate_fn=image_provider.collate_fn)
    
    return train_loader, val_loader

class Trainer(object):
    def __init__(self,args,opts):
        self.global_step = 0
        self.epoch = 0
        self.opts = opts
        create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints_edges1_augmented6'))

       # Copy experiment file
        os.system('cp %s %s'%(args.exp, self.opts['exp_dir']))

        #self.writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train'))
        #self.val_writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train_val'))

        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], image_provider.DataProvider)
        self.model = Model(64,64*12,3)
        self.model = self.model.to(device)
        #self.model.edge_model.reload(self.opts["model_path"])
        
        self.edge_loss_fn =nn.MSELoss()
        #self.fp_loss_fn = nn.BCELoss()
        # Allow individual options
        wd = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue
            else:
                wd.append(p)
                print(name)
        self.optimizer = optim.Adam([
                {'params': wd}
            ],lr = self.opts['lr'])
        for name,param in self.model.named_parameters():
            print(name)
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'], 
            gamma=0.1)
       
    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints_edges1_augmented6', 'epoch%d_step%d.pth'\
        %(epoch, self.global_step))
        torch.save(save_state, save_name)
        print('Saved model')

    def resume(self, path):
        self.model.reload(path)
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])

        print('Model reloaded to resume from Epoch %d, Global Step %d from model at %s'%(self.epoch, self.global_step, path)) 

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            self.save_checkpoint(epoch)
#            self.lr_decay.step()
#            print 'LR is now: ', self.optimizer.param_groups[0]['lr']
            self.train(epoch)
        
    def train(self, epoch):
        print('Starting training')
        self.model.train()
        edge_losses = []
        fp_losses = []
        losses = []
        accum = defaultdict(float)
        
        for step, data in enumerate(self.train_loader):     
            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()
        #self.model.train()
                self.save_checkpoint(epoch)
            img = data['img']
            img = torch.cat(img)
            img = img.view(-1,64,768,3)
            img = torch.transpose(img,1,3)
            img = torch.transpose(img,2,3)
            img = img.float()
            edge_logits = self.model(img.cuda())
            edge_mask = data["edge_mask"]
            #fp_mask = data["fp_mask"]
            #fp_mask = fp_mask.view(fp_mask.shape[0],-1)
            self.optimizer.zero_grad()
            edge_loss = self.edge_loss_fn(edge_logits,edge_mask.cuda())
            #fp_loss =self.fp_loss_fn(fp_logits,fp_mask.cuda())
            loss = edge_loss 
            loss.backward()
            self.optimizer.step()
            #edge_losses.append(edge_loss.item())
            fp_losses.append(edge_loss.item())
            losses.append(loss.item())
            accum['loss'] += float(loss.item())
            accum['length'] += 1
           
            if(step%self.opts['print_freq']==0):
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                print("[%s] Epoch: %d, Step: %d, Loss: %f"%(str(datetime.now()), epoch, self.global_step, accum['loss']))
                accum = defaultdict(float)
            
            
       
            self.global_step += 1
        avg_epoch_loss = 0.0
        #avg_edge_loss = 0.0
        #avg_fp_loss = 0.0
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            #avg_edge_loss += edge_losses[i]
            #avg_fp_loss += fp_losses[i]
        avg_epoch_loss = avg_epoch_loss/len(losses)
        #avg_edge_loss = avg_edge_loss/len(losses)
       # avg_fp_loss = avg_fp_loss/len(losses)
        print("Average Epoch %d loss is : %f "%(epoch,avg_epoch_loss))
    def validate(self):
        print('Validating')
        self.model.eval()
        edge_losses = []
        fp_losses = []
        losses = []
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                img = data['img']
          
                img = torch.cat(img)
                img = img.view(-1,64,768,3)
                img = torch.transpose(img,1,3)
                img = torch.transpose(img,2,3)
                edge_mask = data["edge_mask"]
                #fp_mask = data["fp_mask"]
                #fp_mask = fp_mask.view(fp_mask.shape[0],-1)
                img = img.float()
           
                edge_logits = self.model(img.cuda())
                edge_loss =self.edge_loss_fn(edge_logits,edge_mask.cuda())
                #fp_loss = self.fp_loss_fn(fp_logits,fp_mask.cuda())
                loss = edge_loss
                losses.append(loss.item())
                #edge_losses.append(edge_loss.item())
                fp_losses.append(edge_loss.item())
            
        avg_epoch_loss = 0.0
        #avg_edge_loss = 0.0
        #avg_fp_loss = 0.0
        for i in range(len(losses)):
            avg_epoch_loss += losses[i]
            #avg_edge_loss += edge_losses[i]
            #avg_fp_loss += fp_losses[i]
        avg_epoch_loss = avg_epoch_loss/len(losses)
        #avg_edge_loss = avg_edge_loss/len(losses)
        #avg_fp_loss = avg_fp_loss/len(losses)
        print("Average VAL error is : %f "%(avg_epoch_loss))
        self.model.train()
if __name__ == '__main__':
    args = get_args()
    opts = json.load(open(args.exp, 'r'))
    trainer = Trainer(args,opts)
    trainer.loop()
