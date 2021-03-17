import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils import data 
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import time
import random

def medi_imread(path):
    
    img_file = sitk.ReadImage(path) # image format load
    #img = sitk.GetArrayFromImage(img_file) #image array load
    
    return img_file

def save_3dimg (save_path,img_file,img):

    seg2_file = sitk.GetImageFromArray(img)
    spacing = img_file.GetSpacing()
    seg2_file.SetSpacing(spacing)
    origin = img_file.GetOrigin()
    seg2_file.SetOrigin(origin)
    direction = img_file.GetDirection()
    seg2_file.SetDirection(direction)
    sitk.WriteImage(seg2_file, save_path)
    
    return "image saved"
def compute_loss_smooth(mat):

  return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
         torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

def saturate_mask(m, saturate=False):
  return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m

def target_fixed(target,val):
  target = torch.zeros_like(target)
  for j in range(target.shape[0]):
    target[j]=val

  return target


def classification_loss(logit, target):
  """Compute binary or softmax cross entropy loss."""
  return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

def update_lr(lr,optimizer_G,optimizer_2D,optimizer_3D):
  """Decay learning rates of the generator and discriminator."""
  for param_group in optimizer_G.param_groups:
  	param_group['lr'] = lr
  for param_group in optimizer_2D.param_groups:
  	param_group['lr'] = lr
  for param_group in optimizer_3D.param_groups:
    param_group['lr'] = lr
  return optimizer_G,optimizer_2D,optimizer_3D

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
      m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
      m.weight.data.normal_(1.0, 0.02)
      m.bias.data.fill_(0)

def gradient_penalty(y, x,device):
  """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
  weight = torch.ones(y.size()).to(device)
  dydx = torch.autograd.grad(outputs=y,
                             inputs=x,
                             grad_outputs=weight,
                             retain_graph=True,
                             create_graph=True,
                             only_inputs=True)[0]

  dydx = dydx.view(dydx.size(0), -1)
  dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
  return torch.mean((dydx_l2norm-1)**2)

class ADNI_MRI(data.Dataset):
    """Dataset class for the ADNI dataset."""

    def __init__(self, image_dir, nserial, mode):
        """Initialize and preprocess the ADNI dataset."""
        self.image_dir = image_dir
        self.mode = mode
        self.dataset = []
        self.nserial = nserial
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the ADNI attribute file."""

        for class_name in ["/AD","/MCI","/NM"]:
          for file_name in os.listdir(self.image_dir+class_name):
              path = self.image_dir +class_name+"/"+ file_name

              if class_name=="/NM":
                  label = [0.0]
              elif class_name=="/MCI":
                  label = [0.5]
              elif class_name=="/AD":
                  label = [1.0]

              self.dataset.append([path, np.array(label), file_name])

        random.seed(1234)
        random.shuffle(self.dataset)
        print('Finished preprocessing the ADNI dataset...')

    def augmentation_2d(self,img_arr):
        ori_size = np.copy(int(img_arr.shape[1]))
        for i in range(img_arr.shape[0]):
            p = np.random.random()
            r_size = int(np.ceil(np.random.uniform(ori_size*0.9,ori_size)))
            h = int(np.ceil(np.random.uniform(0,ori_size-r_size)))       
            w = int(np.ceil(np.random.uniform(0,ori_size-r_size)))
            img_2d = img_arr[i,:,:]  
            if p>0.5:
                img_2d = np.fliplr(img_2d)  

            img_2d = img_2d[h:h+r_size,w:w+r_size]  
            img_2d = Image.fromarray(img_2d)
            img_2d = img_2d.resize((ori_size,ori_size),resample=Image.BICUBIC)
            img_2d = np.array(img_2d)
            img_2d = (img_2d/127.5)-1.
            img_2d = img_2d + np.abs(np.min(img_2d)+1)
            img_arr[i,:,:]=img_2d
        img_arr = img_arr[:,np.newaxis,:,:]
        return torch.FloatTensor(img_arr)

    def augmentation_3d(self,img_arr):
        p = np.random.random()
        ori_size = int(img_arr.shape[1])
        r_size = int(np.ceil(np.random.uniform(ori_size*0.9,ori_size)))
        h = int(np.ceil(np.random.uniform(0,ori_size-r_size)))       
        w = int(np.ceil(np.random.uniform(0,ori_size-r_size)))  

        if p>0.5:
            for i in range(img_arr.shape[0]):
                img_2d = img_arr[i,:,:]
                img_2d = np.fliplr(img_2d)
                img_arr[i,:,:] = img_2d 

        for i in range(img_arr.shape[0]):
            img_2d = img_arr[i,:,:]
            img_2d = img_2d[h:h+r_size,w:w+r_size]  
            img_2d = Image.fromarray(img_2d)
            img_2d = img_2d.resize((ori_size,ori_size),resample=Image.BICUBIC)
            img_2d = np.array(img_2d)
            img_arr[i,:,:] = img_2d

        img_arr = (img_arr/127.5)-1.
        img_arr = img_arr + np.abs(np.min(img_arr)+1)
        return torch.FloatTensor(img_arr)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label.""" 
        path, label, filename = self.dataset[index]
        image = np.copy(np.load(path,mmap_mode="r")) #image size:160*192*192

        if self.mode =="train":
            seed = int((time.time()+index)%(len(self.dataset)*image.shape[0]))
            np.random.seed(seed)     
            s = int(np.ceil(np.random.uniform(20,image.shape[0]-20-1))) # choose middle slice to remove background slice 
            image = image[s:s+self.nserial,:,:] # choose consecutive slice
            image2 = np.copy(image)

            aug2d_img = self.augmentation_2d(image) # data augmentation with 2D unit
            aug3d_img = self.augmentation_3d(image2) # data augmentation with 3D unit         

            label_out = torch.FloatTensor(torch.zeros((self.nserial,1))) # make label for consecutive slices                       
            for i in range(self.nserial):
                label_out[i,:] = torch.FloatTensor(label)

            return aug2d_img, label_out, aug3d_img

        elif self.mode=="val":
            image = (image/127.5)-1.
            image = image[100,:,:]
            return torch.FloatTensor(image[np.newaxis,:,:]), torch.FloatTensor(label)    

        elif self.mode=="test":
            image = (image/127.5)-1.
            image_out = torch.FloatTensor(torch.zeros((image.shape[0],1,image.shape[1],image.shape[2])))
            for i in range(image.shape[0]):
                image_out[i,0,:,:] = torch.FloatTensor(image[i,:,:])

            label_out = torch.FloatTensor(torch.zeros((image.shape[0],1)))           
            for i in range(image.shape[0]):
                label_out[i] = torch.FloatTensor(label)

            return image_out, label_out, filename, self.num_images

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, nserial, image_size=128, 
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""


    dataset = ADNI_MRI(image_dir, nserial, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader