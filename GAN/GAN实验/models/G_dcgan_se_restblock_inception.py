import torch
import torch.nn as nn
import torch.nn.parallel
import  torch.nn.functional as F

# o=s(i-1)+2p-k+2 反卷积



class inception(nn.Module):
    def __init__(self, num_channels):
        super(inception, self).__init__()

        strides = 1
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(num_channels,num_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(3*num_channels,num_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self, input):
        X = self.conv1(input)
        Y = self.conv3(input)
        Z = self.conv5(input)
        C = torch.cat((X, Y), dim=1)
        D = torch.cat((C, Z), dim=1)
        E = self.conv(D)
        return E

class resblock(nn.Module):
    def __init__(self, num_channels):
        super(resblock, self).__init__()

        strides = 1
        self.resblock = nn.Sequential(
            # dconv(num_channels,num_channels,k=3,s=1,p=1),
            nn.ConvTranspose2d(num_channels, num_channels, 3, 1, 1),
            # nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            # dconv(num_channels,num_channels,k=3,s=1,p=1),
            # nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(num_channels, num_channels, 3, 1, 1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self, input):
        Y = self.resblock(input)
        X = input
        return X + Y


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)

        out = self.excitation(out)
        return x*torch.sigmoid(out)


class DCGAN_D(nn.Module):

    # nc=3 nz=100 ndf=ngf=64 isize=64
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu

        # O=(I-K+2P)/S+1;

        assert isize % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential()
        # input is nc x isize x isize

        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        '''
        64*64*3
        32*32*64
        '''

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        # 32
        while csize > 4:
            in_feat = cndf  # 64
            out_feat = cndf * 2  # 128
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))


            cndf = cndf * 2
            csize = csize / 2

        '''
        16*16*128
        cndf=128 csize=16

        8*8*256
        cndf=256 csize=8

        4*4*512
        cndf=512 csize=4

        '''

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):

        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)




class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):

        # nc=3 nz=100 ndf=ngf=64 isize=64
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4

        # cngf = 32  tisize=4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        # cngf = 512

        # input is Z, going into a convolution
        # o=s(i-1)+2p-k+2 反卷积
        '''
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False)) # k s p
    
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))
        '''

        conv1 = nn.Sequential()
        conv1.add_module('initial:{0}-{1}:convt'.format(nz, cngf),nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        conv1.add_module('initial:{0}:batchnorm'.format(cngf),nn.BatchNorm2d(cngf))
        conv1.add_module('initial:{0}:relu'.format(cngf),nn.ReLU(True))

        #ration=4

        se_inception1 = nn.Sequential()
        se_inception1.add_module('inception:{0}-{1}'.format(cngf,cngf),inception(cngf))
        se_inception1.add_module('semodule:{0}-{1}'.format(cngf,cngf),SE(cngf,4))

        restblock1 = nn.Sequential()
        restblock1.add_module('restblock:{0}-{1}'.format(cngf,cngf),resblock(cngf))

        '''
        main.add_module('inception:{0}-{1}'.format(cngf,cngf),inception(cngf))
        main.add_module('semodule:{0}-{1}'.format(cngf,cngf),SE(cngf,4))
        '''






        csize, cndf = 4, cngf

        # csize = 4 cndf = 512

        conv2 = nn.Sequential()
        conv2.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
        conv2.add_module('pyramid:{0}:batchnorm'.format(cngf // 2),nn.BatchNorm2d(cngf // 2))
        conv2.add_module('pyramid:{0}:relu'.format(cngf // 2),nn.ReLU(True))

        se_inception2 = nn.Sequential()
        se_inception2.add_module('inception:{0}-{1}'.format(cngf//2,cngf//2),inception(cngf//2))
        se_inception2.add_module('semodule:{0}-{1}'.format(cngf//2,cngf//2),SE(cngf//2,4))

        restblock2 = nn.Sequential()
        restblock2.add_module('restblock:{0}-{1}'.format(cngf//2,cngf//2),resblock(cngf//2))

        cngf =cngf//2
        csize =csize*2




        conv3 = nn.Sequential()
        conv3.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2), nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
        conv3.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
        conv3.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))

        se_inception3 = nn.Sequential()
        se_inception3.add_module('inception:{0}-{1}'.format(cngf // 2, cngf // 2), inception(cngf // 2))
        se_inception3.add_module('semodule:{0}-{1}'.format(cngf // 2, cngf // 2), SE(cngf // 2, 4))

        restblock3 = nn.Sequential()
        restblock3.add_module('restblock:{0}-{1}'.format(cngf // 2, cngf // 2), resblock(cngf // 2))

        cngf = cngf // 2
        csize = csize * 2


        conv4 = nn.Sequential()
        conv4.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
        conv4.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
        conv4.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))

        se_inception4 = nn.Sequential()
        se_inception4.add_module('inception:{0}-{1}'.format(cngf // 2, cngf // 2), inception(cngf // 2))
        se_inception4.add_module('semodule:{0}-{1}'.format(cngf // 2, cngf // 2), SE(cngf // 2, 4))

        restblock4 = nn.Sequential()
        restblock4.add_module('restblock:{0}-{1}'.format(cngf // 2, cngf // 2), resblock(cngf // 2))

        cngf = cngf // 2
        csize = csize * 2
        '''
        while csize < isize // 2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2),
                            nn.ReLU(True))
            main.add_module('inception:{0}-{1}'.format(cngf//2,cngf//2),inception(cngf//2))
            main.add_module('semodule:{0}-{1}'.format(cngf//2,cngf//2),SE(cngf//2,4))



            cngf = cngf // 2
            csize = csize * 2
        '''

        # Extra layers

        main = nn.Sequential()
        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())

        self.conv1 = conv1
        self.resblock1 = restblock1
        self.se_inception1 = se_inception1

        self.conv2 = conv2
        self.resblock2 = restblock2
        self.se_inception2 =se_inception2


        self.conv3 = conv3
        self.resblock3 = restblock3
        self.se_inception3 = se_inception3

        self.conv4 = conv4
        self.resblock4 =restblock4
        self.se_inception4 =se_inception4
        self.main = main


    def forward(self, input):


        '''
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        '''
        out =self.conv1(input)
        out =self.resblock1(out)+self.se_inception1(out)

        out =self.conv2(out)
        out =self.resblock2(out)+self.se_inception2(out)

        out = self.conv3(out)
        out = self.resblock3(out) + self.se_inception3(out)

        out = self.conv4(out)
        out = self.resblock4(out) + self.se_inception4(out)

        out =self.main(out)



        return out
    ###############################################################################


if __name__ == "__main__":
     G=DCGAN_G(64,100,3,64,1)
     print(G)

