import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from torch.nn import Softmax
from torch.nn import Parameter
import math


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim // 8,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,
                                  out_channels=in_dim // 8,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( N X C X H X W)
        returns :
            out : attention value + input feature
            attention: N X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        # B => N, C, HW
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)
        # B' => N, HW, C
        proj_query = proj_query.permute(0, 2, 1)

        # C => N, C, HW
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # B'xC => N, HW, HW
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(B'xC) => N, HW, HW
        attention = self.softmax(energy)

        # D => N, C, HW
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        # DxS' => N, C, HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # N, C, H, W
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(CAM_Module, self).__init__()

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( N X C X H X W)
        returns :
            out : attention value + input feature
            attention: N X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # N, C, C, bmm 批次矩阵乘法
        energy = torch.bmm(proj_query, proj_key)

        # 这里实现了softmax用最后一维的最大值减去了原始数据, 获得了一个不是太大的值
        # 沿着最后一维的C选择最大值, keepdim保证输出和输入形状一致, 除了指定的dim维度大小为1
        energy_new = torch.max(energy, -1, keepdim=True)
        energy_new = energy_new[0].expand_as(energy)  # 复制的形式扩展到energy的尺寸
        energy_new = energy_new - energy
        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


# def INF(B, H, W):
#     return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
#
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)

        out = self.excitation(out)
        return x * torch.sigmoid(out)


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
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(3 * num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.se = SE(num_channels, 4)

    def forward(self, input):
        X = self.conv1(input)
        Y = self.conv3(input)
        Z = self.conv5(input)
        C = torch.cat((X, Y), dim=1)
        D = torch.cat((C, Z), dim=1)
        E = self.conv(D)
        return E + self.se(input)


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
        self.se = SE(num_channels, 4)

    def forward(self, input):
        Y = self.resblock(input)
        X = input
        return X + Y + self.se(input)

class block(nn.Module):
    def __init__(self,num_channels):
        super(block, self).__init__()

        self.res = resblock(num_channels)
        self.inception = inception(num_channels)

        self.conv=nn.Sequential(
            nn.Conv2d(2*num_channels,num_channels,1,1,0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
    def forward(self,input):

        output = torch.cat((self.res(input),self.inception(input)),dim=1)
        output = self.conv(output)
        return output



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian


class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        # nc=3 nz=100 ndf=ngf=64 isize=64
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.block =block(ngf*8)

        self.cam=CAM_Module()
        self.ca=CrissCrossAttention(ngf*8)
        #self.pam=PAM_Module(ngf*8)
        self.main2=nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )



    def forward(self,input):

        #print(input.shape)
        out=self.main(input)
        out=self.block(out)
        out=self.cam(out)+self.ca(out)
        return self.main2(out)


    #     assert isize % 16 == 0, "isize has to be a multiple of 16"
    #
    #     cngf, tisize = ngf // 2, 4
    #
    #     # cngf = 32  tisize=4
    #     while tisize != isize:
    #         cngf = cngf * 2
    #         tisize = tisize * 2
    #     # cngf = 512
    #
    #     # input is Z, going into a convolution
    #     # o=s(i-1)+2p-k+2 反卷积
    #     '''
    #     main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
    #                     nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False)) # k s p
    #
    #     main.add_module('initial:{0}:batchnorm'.format(cngf),
    #                     nn.BatchNorm2d(cngf))
    #     main.add_module('initial:{0}:relu'.format(cngf),
    #                     nn.ReLU(True))
    #     '''
    #
    #     conv1 = nn.Sequential()
    #     conv1.add_module('initial:{0}-{1}:convt'.format(nz, cngf), nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
    #     conv1.add_module('initial:{0}:batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
    #     conv1.add_module('initial:{0}:relu'.format(cngf), nn.ReLU(True))
    #
    #     # self.inception1 = inception(cngf)
    #     # self.resblock1 = resblock(cngf)
    #     # self.ca1 = CrissCrossAttention(cngf)
    #     self.cam = CAM_Module()
    #     self.pam = PAM_Module(cngf)
    #     # self.sa1 =Self_Attn(cngf)
    #
    #     # ration=4
    #
    #     csize, cndf = 4, cngf
    #
    #     # csize = 4 cndf = 512
    #
    #     conv2 = nn.Sequential()
    #     conv2.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
    #                      nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
    #     conv2.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
    #     conv2.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))
    #
    #     cngf = cngf // 2
    #     csize = csize * 2
    #
    #     conv3 = nn.Sequential()
    #     conv3.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
    #                      nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
    #     conv3.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
    #     conv3.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))
    #
    #     cngf = cngf // 2
    #     csize = csize * 2
    #
    #     conv4 = nn.Sequential()
    #     conv4.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
    #                      nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
    #     conv4.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
    #     conv4.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))
    #
    #     cngf = cngf // 2
    #     csize = csize * 2
    #
    #
    #     # Extra layers
    #
    #     main = nn.Sequential()
    #     main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
    #                     nn.ConvTranspose2d(cngf, nc, 4, 1, 0, bias=False))
    #     main.add_module('final:{0}:tanh'.format(nc),
    #                     nn.Tanh())
    #
    #     self.conv1 = conv1
    #     self.conv2 = conv2
    #     self.conv3 = conv3
    #     self.conv4 = conv4
    #     self.main = main
    #
    # def forward(self, input):
    #     '''
    #     if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
    #         output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    #     else:
    #         output = self.main(input)
    #     '''
    #     out = self.conv1(input)
    #     # out = self.pam(out)+self.cam(out)
    #     # out = self.resblock1(out) + self.inception1(out)
    #     # out = self.ca1(out)
    #
    #     out = self.conv2(out)
    #
    #     out = self.conv3(out)
    #
    #     # out = self.ca3(out)
    #
    #     out = self.conv4(out)
    #
    #     out = self.main(out)
    #     print(out.shape)
    #
    #
    #     return out
    # ###############################################################################


class DCGAN_D(nn.Module):

    # nc=3 nz=100 ndf=ngf= isize=64
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        output= self.main(input)
        #print(output.shape)
        output = output.mean(0)

        return output.view(1)


        # O=(I-K+2P)/S+1;

    #     assert isize % 16 == 0, "isize has to be a multiple of 16"
    #     main = nn.Sequential()
    #     # input is nc x isize x isize
    #
    #     main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
    #                     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
    #     main.add_module('initial:{0}:relu'.format(ndf),
    #                     nn.LeakyReLU(0.2, inplace=True))
    #     csize, cndf = isize / 2, ndf
    #
    #     while csize > 4:
    #         in_feat = cndf  # 64
    #         out_feat = cndf * 2  # 128
    #         main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
    #                         nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
    #         main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
    #                         nn.BatchNorm2d(out_feat))
    #         main.add_module('pyramid:{0}:relu'.format(out_feat),
    #                         nn.LeakyReLU(0.2, inplace=True))
    #         cndf = cndf * 2
    #         csize = csize / 2
    #
    #     # state size. K x 4 x 4
    #     main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
    #                     nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))# 4x4
    #     main.add_module(nn.Sigmoid())
    #     self.main = main
    #
    # def forward(self, input):
    #
    #     if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
    #         output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    #     else:
    #         output = self.main(input)
    #
    #
    #     #print(output.shape)
    #
    #
    #
    #     output = output.mean()
    #     return output.view(1)
