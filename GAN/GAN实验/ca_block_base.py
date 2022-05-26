import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from torch.nn import Softmax
from torch.nn import  Parameter
import math



# def INF(B, H, W):
#     return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
#
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

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


#
#
# class GCN(nn.Module):
#     def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
#         super(GCN, self).__init__()
#         self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
#         self._num_nodes = adj.shape[0]
#         self._input_dim = input_dim  # seq_len for prediction
#         self._output_dim = output_dim  # hidden_dim for prediction
#         self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('tanh'))
#
#     def forward(self, inputs):
#         # (batch_size, seq_len, num_nodes)
#         batch_size = inputs.shape[0]
#         # (num_nodes, batch_size, seq_len)
#         inputs = inputs.transpose(0, 2).transpose(1, 2)
#         # (num_nodes, batch_size * seq_len)
#         inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
#         # AX (num_nodes, batch_size * seq_len)
#         ax = self.laplacian @ inputs
#         # (num_nodes, batch_size, seq_len)
#         ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
#         # (num_nodes * batch_size, seq_len)
#         ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
#         # act(AXW) (num_nodes * batch_size, output_dim)
#         outputs = torch.tanh(ax @ self.weights)
#         # (num_nodes, batch_size, output_dim)
#         outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
#         # (batch_size, num_nodes, output_dim)
#         outputs = outputs.transpose(0, 1)
#         return outputs
#
#     @staticmethod
#     def add_model_specific_arguments(parent_parser):
#         parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument('--hidden_dim', type=int, default=64)
#         return parser
#
#     @property
#     def hyperparameters(self):
#         return {
#             'num_nodes': self._num_nodes,
#             'input_dim': self._input_dim,
#             'output_dim': self._output_dim
#         }


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
        conv1.add_module('initial:{0}-{1}:convt'.format(nz, cngf), nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        conv1.add_module('initial:{0}:batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        conv1.add_module('initial:{0}:relu'.format(cngf), nn.ReLU(True))

        # ration=4

        self.inception = inception(cngf)
        self.resblock = resblock(cngf)
        self.cross_attention = CrissCrossAttention(cngf)

        csize, cndf = 4, cngf

        # csize = 4 cndf = 512

        conv2 = nn.Sequential()
        conv2.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                         nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
        conv2.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
        conv2.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))

        cngf = cngf // 2
        csize = csize * 2

        conv3 = nn.Sequential()
        conv3.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                         nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
        conv3.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
        conv3.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))

        cngf = cngf // 2
        csize = csize * 2

        self.ca = CrissCrossAttention(cngf)

        conv4 = nn.Sequential()
        conv4.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                         nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
        conv4.add_module('pyramid:{0}:batchnorm'.format(cngf // 2), nn.BatchNorm2d(cngf // 2))
        conv4.add_module('pyramid:{0}:relu'.format(cngf // 2), nn.ReLU(True))

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
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.main = main

    def forward(self, input):
        '''
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        '''
        out = self.conv1(input)
        out = self.resblock(out) + self.inception(out)
        out = self.cross_attention(out)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.ca(out)

        out = self.conv4(out)

        out = self.main(out)

        return out
    ###############################################################################


class DCGAN_D(nn.Module):

    # nc=3 nz=100 ndf=ngf= isize=64
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

        # 128*128*3 -> 64*64*64
        # csize = 32

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


class DCGAN_D_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:conv'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

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


class DCGAN_G_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
