from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
import  wgan
import sagan
import cagan
import ca_block_gan
import models.dcgan


from torchsummary import summary
import  wgan



if __name__=="__main__":

    imageSize=64
    # nc=3 nz=100 ndf=ngf=64 isize=64
    nz=100
    nc=3
    ngf=64
    ngpu=1
    #
    #
    #
    image_num = 100
    # #print(netG)
    #
    # # load weights
    #
    # #
    # netG = wgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu)
    # path = 'netG_epoch_1200.pth'
    #
    #
    # netG.load_state_dict(torch.load(path))
    # netG.cuda()
    # summary(netG,input_size=(100,1,1))
    #
    # fixed_noise = torch.FloatTensor(image_num, nz, 1, 1).normal_(0, 1)
    # fixed_noise = fixed_noise.cuda()
    #
    # fake = netG(fixed_noise)
    # fake.data = fake.data.mul(0.5).add(0.5)






    Species = ['healthy','leaf_mold','leaf_curl','spider_mite']

    for species in Species:

        load_path = species + '1200'
        os.mkdir(load_path)

        path = species+'/netG_epoch_'+'1200.pth'




        netG = wgan.DCGAN_G(imageSize, nz, nc, ngf, ngpu)


        netG.load_state_dict(torch.load(path))
        netG.cuda()

        fixed_noise = torch.FloatTensor(image_num, nz, 1, 1).normal_(0, 1)
        fixed_noise = fixed_noise.cuda()

        fake = netG(fixed_noise)
        fake.data = fake.data.mul(0.5).add(0.5)

        print(path)
        for i in range(image_num):
            print('Generate image......')
            vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(load_path, "%d.png"%i))


# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#    ConvTranspose2d-1            [-1, 512, 4, 4]         819,200
#        BatchNorm2d-2            [-1, 512, 4, 4]           1,024
#               ReLU-3            [-1, 512, 4, 4]               0
#             Conv2d-4             [-1, 64, 4, 4]          32,832
#             Conv2d-5             [-1, 64, 4, 4]          32,832
#            Softmax-6               [-1, 16, 16]               0
#             Conv2d-7            [-1, 512, 4, 4]         262,656
#          Self_Attn-8            [-1, 512, 4, 4]               0
#    ConvTranspose2d-9            [-1, 256, 8, 8]       2,097,152
#       BatchNorm2d-10            [-1, 256, 8, 8]             512
#              ReLU-11            [-1, 256, 8, 8]               0
#   ConvTranspose2d-12          [-1, 128, 16, 16]         524,288
#       BatchNorm2d-13          [-1, 128, 16, 16]             256
#              ReLU-14          [-1, 128, 16, 16]               0
#   ConvTranspose2d-15           [-1, 64, 32, 32]         131,072
#       BatchNorm2d-16           [-1, 64, 32, 32]             128
#              ReLU-17           [-1, 64, 32, 32]               0
#            Conv2d-18            [-1, 8, 32, 32]             520
#            Conv2d-19            [-1, 8, 32, 32]             520
#           Softmax-20           [-1, 1024, 1024]               0
#            Conv2d-21           [-1, 64, 32, 32]           4,160
#         Self_Attn-22           [-1, 64, 32, 32]               0
#   ConvTranspose2d-23            [-1, 3, 64, 64]           3,072
#              Tanh-24            [-1, 3, 64, 64]               0
# ================================================================
# Total params: 3,910,224
# Trainable params: 3,910,224
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 12.27
# Params size (MB): 14.92
# Estimated Total Size (MB): 27.18
# ----------------------------------------------------------------