import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os.path as osp
import cv2
import json
import multiprocessing.dummy as multiprocessing
from skimage.io import imread
from skimage.transform import resize
import skimage.color as color
from skimage.io import imsave
import utils
import torchvision
import random
EPS = 1e-7 
def process_info(args):
    """
    Process a single json file
    """
    fname, opts = args
    
    with open(fname, 'r') as f:
        ann = json.load(f)
        f.close()
    examples = []
    skipped_instances = 0

    for instance in ann:
        components = instance['components']

        if 'class_filter'in opts.keys() and instance['label'] not in opts['class_filter']:
            continue
        
        candidates = [c for c in components if len(c['poly']) >= opts['min_poly_len']]

        if 'sub_th' in opts.keys():
            total_area = np.sum([c['area'] for c in candidates])
            candidates = [c for c in candidates if c['area'] > opts['sub_th']*total_area]

        candidates = [c for c in candidates if c['area'] >= opts['min_area']]

        if opts['skip_multicomponent'] and len(candidates) > 1:
            skipped_instances += 1
            continue

        instance['components'] = candidates
        if candidates:
            examples.append(instance)

    return examples, skipped_instances   

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    #Samples elements randomly from a given list of indices for imbalanced dataset
    def __init__(self, dataset,weight, indices=None, num_samples=None):
      
         # if indices is not provided, 
        # all elements in the dataset will be considered
        if indices is None:
            self.indices = list(range(len(dataset))) 
        else:
            self.indices = indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        if num_samples is None:
            self.num_samples = len(self.indices) 
        else:
            self.num_samples = num_samples
            
        # weight for each sample
       
        self.weights = torch.DoubleTensor(weight)
        
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=False).tolist())
    def __len__(self):
        return self.num_samples

def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}

    for key in keys:
        val = [item[key] for item in batch_list]

        t = type(batch_list[0][key])
        
        if t is np.ndarray:
            try:
                val = torch.from_numpy(np.stack(val, axis=0))
            except:
                # for items that are not the same shape
                # for eg: orig_poly
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated

class DataProvider(Dataset):
    """
    Class for the data provider
    """
    def __init__(self, opts, split='train', mode='train'):
        """
        split: 'train', 'train_val' or 'val'
        opts: options from the json file for the dataset
        """
        self.opts = opts
        self.mode = mode
        self.label_to_count = {}
        self.weight = []
        print('Dataset Options: ', opts)
        if self.mode !='tool':
            # in tool mode, we just use these functions
            self.data_dir = osp.join(opts['data_dir'], split)
            self.instances = []
            self.read_dataset()
            print('Read %d instances in %s split'%(len(self.instances), split))
    def read_dataset(self):
        data_list = glob.glob(osp.join(self.data_dir, '*/*.json'))
        data_list = [[d, self.opts] for d in data_list]
        print(len(data_list))
        pool = multiprocessing.Pool(self.opts['num_workers'])
        data = pool.map(process_info, data_list)
        pool.close()
        pool.join()
        print(len(data))

        print("Dropped %d multi-component instances"%(np.sum([s for _,s in data])))
        
        self.instances = [instance for image,_ in data for instance in image]
        self.instances = self.instances[:-6]
        for instance in self.instances:
            label = instance['label']
            if label not in self.label_to_count:
                self.label_to_count[label]=1
            else:
                self.label_to_count[label]+=1
        #list of weights that are inverse of frequency of that class
        self.weight = [1.0/self.label_to_count[instance['label']] for instance in self.instances] 
        if 'debug' in self.opts.keys() and self.opts['debug']:
            self.instances = self.instances[:16]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.prepare_instance(idx)
    def getweight(self):
        return self.weight
    def getlabeltocount(self):
        return self.label_to_count
    def prepare_instance(self, idx):
        """
        Prepare a single instance, can be both multicomponent
        or just a single component
        """
        instance = self.instances[idx]
        component = instance['components'][0]
        
        results = self.prepare_component(instance, component)
        
        return results

    def prepare_component(self, instance, component):
        img = cv2.imread(instance['image_url'])
        label = instance['label']
        bbox = component['bbox']
        x0 = max(int(bbox[0]),0)
        y0 = max(int(bbox[1]),0)
        w = max(int(bbox[2]),0)
        h = max(int(bbox[3]),0)
        poly = component["poly"]
        
        img = img[y0:y0+h,x0:x0+w]
        color_jitter = torchvision.transforms.ColorJitter(brightness=2.5, contrast=2.5, saturation=2.5)
        grey_scale = torchvision.transforms.Grayscale(num_output_channels=3)
        pil = torchvision.transforms.ToPILImage()
        if self.mode=="train":
            img = pil(img)
            if random.randint(1,4)==1:
                img = color_jitter(img)
            else:
                if random.randint(1,4)==1:
                    img = grey_scale(img)
            img = np.array(img)
        else:
            img2 = pil(img)
            img2 = color_jitter(img2)
            img3 = grey_scale(pil(img))
            img2 = np.array(img2)
            img3 = np.array(img3)
            #if random.randint(1,5)==1:
            # img = cv2.GaussianBlur(img,(3,3),0)
            #print(img.shape)
        poly = np.array(poly).astype(np.float)
        poly[:,0] = poly[:,0] - x0
        poly[:,1] = poly[:,1] - y0
        poly[:,0] = poly[:,0]/float(w)
        poly[:,1] = poly[:,1]/float(h) 
        edge_mask = np.zeros((32,384),np.float32)
     
        edge_mask = utils.get_edge_mask(poly,edge_mask)
        fp_mask = np.zeros((16,192),np.float32)
        fp_mask = utils.get_fp_mask(poly,fp_mask)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #index = 0
       # for i in range(len(instance['image_url'])):
            #if instance['image_url'][i]=='/':
                #index = i
       # imsave("/home/jatin/image_classifier/test_image/"+instance['image_url'][index+1:],img)
        #print("saved image")
        img  = cv2.resize(img,(768,64))
       # img2  = cv2.resize(img2,(768,64))
        #img3  = cv2.resize(img3,(768,64))
        #print(img)
        #imsave("/home/jatin/image_classifier/test_image/"+"resize"+instance['image_url'][index+1:],img)
        #print('out of resize')
        #print(img.shape)
        img = torch.from_numpy(img)

        return_dict = {
                    "img":img,
                    "label":label,
                    "poly" : poly,
                    "edge_mask":edge_mask,
                    "fp_mask":fp_mask,
                    "x0" : x0,
                    "y0" : y0,
                    "w" : w,
                    "h" : h
                   # "img2":img2,
                   # "img3":img3
                    }
        
        return return_dict
