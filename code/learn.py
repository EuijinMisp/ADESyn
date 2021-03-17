import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
from utils import *
import time
import numpy as np

def train(args, netG, net2D, net3D, device):
    ### Model load if pretrained models are used. ###
    if args.retrain == True:
        try:
            netG.load_state_dict(torch.load(args.load_model_G, map_location="cuda:"+str(args.ngpu)))
            net2D.load_state_dict(torch.load(args.load_model_2D, map_location="cuda:"+str(args.ngpu)))
            net3D.load_state_dict(torch.load(args.load_model_3D, map_location="cuda:"+str(args.ngpu)))
            print("pretrained model load done")
        except:
            raise("pretrained model load fail")
    else:
        args.init_epochs=0

    train_loader = get_loader(args.img_dir, args.nserial, args.imageSize, args.batch_size, args.mode, args.workers)
    val_loader = get_loader(args.val_dir, args.nserial, args.imageSize, args.batch_size, "val", args.workers)
    mse_loss = torch.nn.MSELoss().cuda()

    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_2D = optim.Adam(net2D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_3D = optim.Adam(net3D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    ### load fixed image for  validation ###
    dloader = iter(val_loader)
    fixed_img,fixed_target = next(dloader)

    target_NM = target_fixed(fixed_target,0.0).to(device)
    target_MCI = target_fixed(fixed_target,0.5).to(device)
    target_AD = target_fixed(fixed_target,1.0).to(device)

    fixed_img=fixed_img.to(device) 
    print("Input data shape")
    print(fixed_img.shape)
    print(fixed_target.shape)

    ### training start ###
    for epoch in range(args.init_epochs,args.epochs):
        epoch_start=time.time()
        for i in range(args.niter):

            ### Input data load ###
            start = time.time()
            dloader = iter(train_loader)
            img,s_label,img_o3d = next(dloader) #img:[Batch,nserial,1,192,192],s_label:[Batch,nserial,1],img_o3d:[Batch,nserial,192,192]

            ## prepare two set of images using augmentation to learn 2D discriminator and 3D discriminator 
            torch.manual_seed(i+args.niter*epoch)          
            rand_idx = torch.randperm(s_label.size(0))
            t_label = s_label[rand_idx]
            s_label = s_label.view(s_label.size(0)*s_label.size(1),s_label.size(2))
            t_label = t_label.view(t_label.size(0)*t_label.size(1),t_label.size(2))

            img=img.to(device) 
            img_o3d=img_o3d.to(device) 
            s_label = s_label.to(device)
            t_label = t_label.to(device)

            img = img.view(img.size(0)*img.size(1),args.n_ch,img.size(3),img.size(4)) ## img:(36,1,192,192)
            img_o3d = img_o3d.view(args.batch_size,args.n_ch,args.nserial,img_o3d.size(2),img_o3d.size(3)) ##img_o3d:(6,1,6,192,192)
            s_label_3d = s_label.view(args.batch_size,int(s_label.size(0)/args.batch_size),1) ## s_label_3d:(6,6,1)
            s_label_3d = s_label_3d[:,0,:] ##s_label_3d:(6,3)

            t_label_3d = t_label.view(args.batch_size,int(t_label.size(0)/args.batch_size),1)
            t_label_3d = t_label_3d[:,0,:] ##t_label_3d:(6,3)

            #### Discriminator training ####
            ###########################real images to D ########################################
            net3D.zero_grad()
            out_src_3d = net3D(img_o3d)
            real_src_loss_3d = - torch.mean(out_src_3d)

            net2D.zero_grad()
            out_src_2d,out_cls_2d = net2D(img)
            real_src_loss_2d = - torch.mean(out_src_2d)
            real_cls_loss_2d = (mse_loss(out_cls_2d,s_label)/img.size(0))*args.lambda_cls
            
            ###########################real images to G ########################################
            img_o2d = img_o3d.view(img_o3d.size(0)*img_o3d.size(2),args.n_ch,img_o3d.size(3),img_o3d.size(4)) ##(36,1,192,192)
            fake_img_3d, fake_img_3d_mask = netG(img_o2d,t_label)
            fake_img_3d_mask = saturate_mask(fake_img_3d_mask, saturate=True)
            fake_imgs_3d_masked = fake_img_3d_mask * img_o2d + (1 - fake_img_3d_mask) * fake_img_3d
            fake_imgs_3d_masked = fake_imgs_3d_masked.view(img_o3d.size(0),img_o3d.size(1),img_o3d.size(2),img_o3d.size(3),img_o3d.size(4)) ##(36,1,192,192)

            fake_img, fake_img_mask = netG(img,t_label)
            fake_img_mask = saturate_mask(fake_img_mask, saturate=True)
            fake_imgs_masked = fake_img_mask * img + (1 - fake_img_mask) * fake_img

            ###########################fake images to D ########################################
            out_src_3d = net3D(fake_imgs_3d_masked.detach())
            fake_src_loss_3d = torch.mean(out_src_3d)
            alpha_3d = torch.rand(img_o3d.size(0), 1, 1, 1, 1).to(device)

            x_hat_3d = (alpha_3d * img_o3d.data + (1 - alpha_3d) * fake_imgs_3d_masked.data).requires_grad_(True)
            out_src_3d = net3D(x_hat_3d)
            d_loss_gp_3d = gradient_penalty(out_src_3d, x_hat_3d,device)

            out_src_2d,_ = net2D(fake_imgs_masked.detach())
            fake_src_loss_2d = torch.mean(out_src_2d)
            alpha_2d = torch.rand(img.size(0), 1, 1, 1).to(device)

            x_hat_2d = (alpha_2d * img.data + (1 - alpha_2d) * fake_imgs_masked.data).requires_grad_(True)
            out_src_2d, _ = net2D(x_hat_2d)
            d_loss_gp_2d = gradient_penalty(out_src_2d, x_hat_2d,device)
            ##############################################################################
            
            dloss_3d = (real_src_loss_3d+fake_src_loss_3d)+(d_loss_gp_3d*10)
            dloss_2d = (real_src_loss_2d+fake_src_loss_2d)+(real_cls_loss_2d)+(d_loss_gp_2d*10)

            dloss_3d.backward()
            optimizer_3D.step()

            dloss_2d.backward()
            optimizer_2D.step()

            #### Generator training #### 
            if (i+1) % args.n_critic == 0:
                netG.zero_grad()
                fake_img_2d,fake_img_2d_mask = netG(img_o2d,t_label)
                fake_img_2d_mask = saturate_mask(fake_img_2d_mask, saturate=True)
                
                fake_img_3d = fake_img_2d.view(args.batch_size,args.n_ch,args.nserial,fake_img.size(2),fake_img.size(3))
                fake_img_3d_mask = fake_img_2d_mask.view(args.batch_size,args.n_ch,args.nserial,fake_img.size(2),fake_img.size(3))

                fake_imgs_2d_masked = fake_img_2d_mask * img_o2d + (1 - fake_img_2d_mask) * fake_img_2d
                fake_imgs_3d_masked = fake_img_3d_mask * img_o3d + (1 - fake_img_3d_mask) * fake_img_3d

                out_src_2d,out_cls_2d = net2D(fake_imgs_2d_masked)
                out_src_3d = net3D(fake_imgs_3d_masked)

                if args.nserial<3:
                    ratio_3d=0.5
                else:
                    ratio_3d = 1/(args.nserial-1)

                fake_src_loss_2d = - torch.mean(out_src_2d)*(1-ratio_3d)
                fake_cls_loss_2d = (mse_loss(out_cls_2d,t_label)/img.size(0))*args.lambda_cls
                fake_src_loss_3d = - torch.mean(out_src_3d)*ratio_3d

                recon_loss = torch.mean(torch.abs(fake_imgs_2d_masked - img_o2d)*(1 - fake_img_2d_mask))*args.lambda_recon
                mask_loss = torch.mean(fake_img_2d_mask) * args.lambda_mask
                mask_smooth_loss = compute_loss_smooth(fake_img_2d_mask) * args.lambda_mask_smooth

                gloss = (fake_src_loss_3d+fake_src_loss_2d+fake_cls_loss_2d)+(recon_loss)+\
                (mask_loss+mask_smooth_loss)
                
                gloss.backward()
                optimizer_G.step()
                end = time.time()
                print('[%d/%d][%d/%d] Loss_3D: %.4f Loss_2D: %.4f Loss_G: %.4f time: %.2f '
                  % (epoch+1, args.epochs, i, args.niter,
                     dloss_3d.item(), dloss_2d.item(), gloss.item(), end-start))         

        if epoch >= int(args.epochs/2):
            args.lr -= (args.lr / float(args.epochs/2))
            optimizer_G,optimizer_2D,optimizer_3D = update_lr(args.lr,optimizer_G,optimizer_2D,optimizer_3D)
            print ('Decayed learning rates, lr: {}.'.format(args.lr))
        
        with torch.no_grad():
            fake_NM,fake_NM_mask = netG(fixed_img,target_NM)
            fake_MCI,fake_MCI_mask = netG(fixed_img,target_MCI)
            fake_AD,fake_AD_mask = netG(fixed_img,target_AD)

        fake_imgs_NM = fake_NM_mask * fixed_img + (1 - fake_NM_mask) * fake_NM
        fake_imgs_MCI = fake_MCI_mask * fixed_img + (1 - fake_MCI_mask) * fake_MCI
        fake_imgs_AD = fake_AD_mask * fixed_img + (1 - fake_AD_mask) * fake_AD

        print_list1 = [fake_imgs_NM,fake_NM_mask,fake_NM]
        print_list2 = [fake_imgs_MCI,fake_MCI_mask,fake_MCI]
        print_list3 = [fake_imgs_AD,fake_AD_mask,fake_AD]

        vutils.save_image(fixed_img,'%s/real_samples.png' % args.train_out,normalize=True) 
        vutils.save_image(torch.cat(print_list1+print_list2+print_list3, dim=0).detach(),'%s/fake_epoch_%03d.png' % (args.train_out, epoch+1),normalize=True)
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.train_out, epoch+1))
        torch.save(net2D.state_dict(), '%s/net2D_epoch_%d.pth' % (args.train_out, epoch+1))
        torch.save(net3D.state_dict(), '%s/net3D_epoch_%d.pth' % (args.train_out, epoch+1))
        epoch_end=time.time()
        print('1 epoch time: %.2f sec' %(epoch_end-epoch_start))

def test(args, netG, device):
    try:
        netG.load_state_dict(torch.load(args.load_model_G, map_location="cuda:"+str(args.ngpu)))
        print("model load done")
    except:
        raise("model load fail")

    test_loader = get_loader(args.img_dir, args.nserial, args.imageSize, 
               args.batch_size, args.mode, args.workers)               
    medi_format = medi_imread(args.medical_format)

    with torch.no_grad():
        for index,(img,label, fname, n_data) in enumerate(test_loader):
            img = img[0]
            label=label[0]
            target_00 = target_fixed(label,0).to(device)
            target_05 = target_fixed(label,0.5).to(device)
            target_10 = target_fixed(label,1.0).to(device)

            img= img.to(device)
            fake_00,fake_00_mask = netG(img,target_00)
            fake_05,fake_05_mask = netG(img,target_05)
            fake_10,fake_10_mask = netG(img,target_10)

            fake_imgs_00 = fake_00_mask * img + (1 - fake_00_mask) * fake_00
            fake_imgs_05 = fake_05_mask * img + (1 - fake_05_mask) * fake_05
            fake_imgs_10 = fake_10_mask * img + (1 - fake_10_mask) * fake_10

            fake_imgs_00 = fake_imgs_00.cpu().numpy()[:,0,:,:]
            fake_imgs_05 = fake_imgs_05.cpu().numpy()[:,0,:,:]
            fake_imgs_10 = fake_imgs_10.cpu().numpy()[:,0,:,:]
            fake_10_mask = fake_10_mask.cpu().numpy()[:,0,:,:]
            fake_10 = fake_10.cpu().numpy()[:,0,:,:]
            original_img = img.cpu().numpy()[:,0,:,:]
            
            save_3dimg(args.test_out+"/"+fname[0][:-11]+"_00.nii.gz",medi_format,(fake_imgs_00+1)*127.5)                    
            save_3dimg(args.test_out+"/"+fname[0][:-11]+"_05.nii.gz",medi_format,(fake_imgs_05+1)*127.5)                          
            save_3dimg(args.test_out+"/"+fname[0][:-11]+"_10.nii.gz",medi_format,(fake_imgs_10+1)*127.5)                     
            print("printing images [%d/%d]"%(index+1,n_data))