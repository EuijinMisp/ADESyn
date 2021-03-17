from __future__ import print_function
import argparse
import torch
import os
import torch.backends.cudnn as cudnn
from network import *
from learn import *
from utils import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Brain MRI generation')
    parser.add_argument('--mode', default='train', help='train or test')
    parser.add_argument('--retrain', type=int, default=0, help='training start from pretrained model')    
    parser.add_argument('--imageSize', type=int, default=192, help='the height / width of the input image to network')
    parser.add_argument('--n_ch', type=int, default=1, help='the height / width of the input image to network')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',help='number of epochs to train')
    parser.add_argument('--init_epochs', type=int, default=0, metavar='N',help='initial number of epochs to retrain')
    parser.add_argument('--niter', type=int, default=5000, metavar='N',help='number of iteration to train')   
    parser.add_argument('--n_critic', type=int, default=5, metavar='N',help='number to alternative training of G and D')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rat')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed')
    parser.add_argument('--save-model', action='store_true', default=True,help='For Saving the current Model')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--train_out', default='../train', help='folder to output validation images and model checkpoints')
    parser.add_argument('--test_out', default='../test', help='folder to output the test images')
    parser.add_argument('--load_model_G', default='../train/netG_epoch_40.pth', help='G model load path')
    parser.add_argument('--load_model_D', default='../train/netD_epoch_40.pth', help='D model load path')
    parser.add_argument('--img_dir', default='../data/NP_3D/setA', help='training image path')
    parser.add_argument('--val_dir', default='../data/NP_3D/setC', help='validation image path')
    parser.add_argument('--medical_format', default='../data/medi_format/002_S_0619_origin.nii', help='medical image file format')
    parser.add_argument('--lambda_recon', type=float, default=10, help='weight for recon loss. default=10')
    parser.add_argument('--lambda_cls', type=float, default=30, help='weight for regression loss. default=30')
    parser.add_argument('--lambda_mask', type=float, default=1e-1, help='weight for mask loss. default=1e-1')
    parser.add_argument('--lambda_mask_smooth', type=float, default=1e-5, help='weight for total variation of mask. default=1e-5')
    parser.add_argument('--nserial', type=int, default=3, help='the number of k for considering 3D consecutiveness. default=3')

    args = parser.parse_args()   
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.ngpu) if use_cuda else "cpu")
    cudnn.benchmark = True   

    ### Network setting ###
    netG = Generator(img_channel=args.n_ch).to(device) 
    net2D = Discriminator_2D(image_size=args.imageSize,img_channel=args.n_ch).to(device)
    net3D = Discriminator_3D(image_size=args.imageSize,img_channel=args.n_ch).to(device)

    netG.apply(weights_init)
    net2D.apply(weights_init) 
    net3D.apply(weights_init) 

    if args.mode =="train":
        
        if os.path.exists(args.train_out):
            pass
        else:
            os.makedirs(args.train_out)

        train(args, netG, net2D, net3D, device)
    
    elif args.mode =="test":

        if os.path.exists(args.test_out):
            pass
        else:
            os.makedirs(args.test_out)

        args.batch_size=1
        test(args, netG, device)
    

if __name__ == '__main__':
    main()