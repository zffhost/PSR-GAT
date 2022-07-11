from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import time
import math
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

try:
    import utils
except:
    from model import utils


def knn_point1(k, xyz1, xyz2):
    """
       Input:
           k: int32, number of k in k-nn search
           xyz1: (batch_size, ndataset, c) float32 array, input points
           xyz2: (batch_size, npoint, c) float32 array, query points
       Output:
           val: (batch_size, npoint, k) float32 array, L2 distances
           idx: (batch_size, npoint, k) int32 array, indices to input points
       """
    xyz1 = torch.unsqueeze(xyz1, 1)
    xyz2 = torch.unsqueeze(xyz2, 2)
    xyz1 = xyz1 - xyz2
    del xyz2
    dist = torch.sum(xyz1 ** 2, -1)
    del xyz1
    val, idx = torch.topk(-dist, k=k)
    del dist

    return val, idx.int()


def knn_point(k, x):
    x = x.transpose(2, 1)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx.int()


def group(xyz, points, k, dilation=1, use_xyz=False):
    idx = knn_point(k, xyz)
    idx = idx[:, :, 1::dilation].contiguous()
    xyz_trans = xyz.transpose(1, 2).contiguous()

    grouped_xyz = utils.grouping_operation(xyz_trans,
                                           idx.detach())
    if grouped_xyz.shape[1] != xyz.shape[1] or idx.shape[1] != xyz.shape[1]:
        pass
    del xyz_trans
    grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()

    xyz_trans = torch.unsqueeze(xyz, 2)

    grouped_xyz -= xyz_trans
    del xyz_trans
    if points is not None:
        grouped_points = utils.grouping_operation(points, idx.detach())
        if use_xyz:
            grouped_points = torch.cat([grouped_xyz, grouped_points], -1)
    else:
        grouped_points = grouped_xyz
    if grouped_points.shape[1] != xyz.shape[1] or idx.shape[1] != xyz.shape[1]:
        pass

    return grouped_xyz, grouped_points, idx


class pointcnn(nn.Module):
    def __init__(self, k, n_cout, n_blocks, bn_decay=None, activation=nn.ReLU(inplace=True)):
        super(pointcnn, self).__init__()
        self.npoint = k
        self.n_blocks = n_blocks

        self.conv1 = nn.Conv2d(3, n_cout, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.conv_list = nn.ModuleList(nn.Conv2d(n_cout, n_cout, kernel_size=(1, 1), stride=(1, 1), bias=False) for i in
                                       range(n_blocks - 1))
        self.relu = activation

    def forward(self, xyz, use_bn=True, use_ibn=False, activation=nn.ReLU(inplace=True)):
        _, grouped_points, _ = group(xyz.detach(), None, self.npoint)
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

        grouped_points = self.relu(self.conv1(grouped_points))
        res = grouped_points  # new

        for idx, c in enumerate(self.conv_list):
            grouped_points = c(grouped_points)

            if idx == self.n_blocks - 2:
                grouped_points = torch.max(grouped_points, 3, keepdim=False,
                                           out=None)

                return grouped_points[0]
            else:
                if use_bn or use_ibn:
                    print('do not use_bn or use_ibn!')
                grouped_points = self.relu(grouped_points)
        grouped_points = torch.cat((grouped_points, res), 2)  # new
        return grouped_points


class res_GAT_block(nn.Module):
    def __init__(self, n_cout, bn_decay=None, use_bn=False, need_conv=True):
        super(res_GAT_block, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.need_conv = need_conv
        if need_conv:
            self.conv1 = nn.Conv2d(n_cout, n_cout, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.conv2 = nn.Conv2d(n_cout, n_cout, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.W = nn.Parameter(torch.zeros(size=(n_cout, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, points, indices=None, k=8):
        shortcut = points  # points:[B,C,N]
        features = points
        feature = points.transpose(2, 1).detach()
        indices = knn_point(k, feature)
        del feature
        features = self.relu(features)
        grouped_points = utils.grouping_operation(features, indices.detach())  # [B,C,N,K]
        center_points = torch.unsqueeze(features, 3)  # [B,C,N,1]
        features = torch.cat([grouped_points, center_points], 3)  # [B,C,N,K+1]
        features = features.permute(0, 2, 3, 1).contiguous()  # [B,N,K+1,C]
        attention = torch.matmul(features, self.W)  # [B,N,K+1,C]->[B,N,K+1,1]
        attention = torch.nn.functional.softmax(attention, dim=2)
        attention = attention.repeat(1, 1, 1, 9)  # [B,N,K+1,1]->[B,N,K+1,K+1]
        features = torch.matmul(attention, features)  # [B,N,K+1,K+1]*[B,N,K+1,C]->[B,N,K+1,C]
        features = features.permute(0, 3, 1, 2).contiguous()  # [B,C,N,K+1]
        features = torch.mean(features, 3, keepdim=False) + shortcut  # [B,C,N]

        if self.need_conv:
            '''
            features = self.conv1(center_points)
            grouped_points_nn = self.conv2(grouped_points)
            features = torch.cat([features, grouped_points_nn], 3)
            features = torch.mean(features, 3, keepdim=False) + shortcut
            '''
            grouped_points_nn = self.conv2(grouped_points)
        del shortcut, attention
        if self.need_conv:
            return features, center_points, grouped_points, grouped_points_nn
        return None, center_points, grouped_points, None


from scipy.stats import truncnorm


def truncated_normal_(tensor, mean=0.0, std=1.0):
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias.data, 0.0)
    return module


class ResLinear(nn.Module):
    def __init__(self, n_in, n_out, n_=32, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), training=False):
        super(ResLinear, self).__init__()
        self.act = act
        self.fc11 = nn.Linear(n_in, n_)
        self.fc12 = nn.Linear(n_, n_out)
        self.fcskip = nn.Linear(n_in, n_out)
        if training:
            fc_init_(self.fc11)
            fc_init_(self.fc12)
            fc_init_(self.fcskip)
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, scale_tensor):
        out_nonlinear = self.fc12(self.act(self.fc11(scale_tensor)))
        out_skip = self.fcskip(scale_tensor)
        return out_nonlinear + out_skip + scale_tensor if self.n_in == self.n_out else out_nonlinear + out_skip


class conv2d_1x1_fixio(nn.Module):
    def __init__(self, base_inp=128, base_oup=128, stride=1, multi_gpu=False, up_scales=np.arange(1.1, 9.1, 0.1),
                 training=True):
        super(conv2d_1x1_fixio, self).__init__()
        self.stride = stride
        self.multi_gpu = multi_gpu
        assert stride in [1, 2]
        max_overall_scale = math.sqrt(up_scales[-1])
        temp = torch.zeros([len(up_scales), 1], requires_grad=False)
        for scale_idx in range(len(up_scales)):
            round_sqrt_up_scale = math.sqrt(up_scales[scale_idx])
            temp[scale_idx][0] = torch.FloatTensor([round_sqrt_up_scale / max_overall_scale])
        self.register_buffer('scale_tensors', temp)
        del temp, round_sqrt_up_scale, max_overall_scale
        self.max_inp_channel = base_inp
        self.max_oup_channel = base_oup
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.skipfc2 = ResLinear(1, self.max_oup_channel * self.max_inp_channel * 1 * 1, n_=32, act=self.activation,
                                 training=training)

    def forward(self, x, inp_scale, mask_t=None):
        x = self.activation(x)
        scale_tensor = self.scale_tensors[int(((inp_scale ** 2) - 1.1) / 0.1)]
        conv1_weight = self.skipfc2(scale_tensor).view(self.max_oup_channel, self.max_inp_channel, 1, 1)
        reg_loss = 0.5 * torch.squeeze(torch.norm(torch.norm(conv1_weight, p=2, dim=0), p=2, dim=0), 0) ** 2
        if mask_t is not None:
            x += self.metamask(mask_t.view(1, 3, -1, 1).contiguous())

        out = F.conv2d(x, conv1_weight[:, :, :, :], bias=None, stride=self.stride, padding=0)
        del conv1_weight, scale_tensor, x
        return out, reg_loss


class res_GAT_meta_block(nn.Module):
    def __init__(self, n_cout, multi_gpus=False, base_inp=128, base_oup=128, up_scales=np.arange(1.1, 9.1, 0.1),
                 training=True, k=0):
        super(res_GAT_meta_block, self).__init__()
        self.k = k
        self.n_cout = n_cout
        self.multi_gpus = multi_gpus
        self.relu = nn.ReLU(inplace=True)
        self.metaconv1 = conv2d_1x1_fixio(multi_gpu=self.multi_gpus, base_inp=base_inp, base_oup=base_oup,
                                          up_scales=up_scales, training=training)
        self.metaconv2 = conv2d_1x1_fixio(multi_gpu=self.multi_gpus, base_inp=base_inp, base_oup=base_oup,
                                          up_scales=up_scales, training=training)

    def forward(self, points, up_scale, indices=None, xyz=None, mask_t=None):
        shortcut = points
        features = points
        if indices is None:
            _, grouped_points, self.indices = group(xyz.detach(), features, self.k)
        else:
            self.indices = indices.detach()
            grouped_points = utils.grouping_operation(features,
                                                      self.indices)
        center_points = torch.unsqueeze(features, 3)
        features, reg_loss = self.metaconv1(center_points, up_scale, mask_t)
        grouped_points_nn, reg_loss2 = self.metaconv2(grouped_points, up_scale, mask_t)
        features = torch.cat([features, grouped_points_nn], 3)
        features = torch.mean(features, 3, keepdim=False) + shortcut
        reg_loss = reg_loss + reg_loss2
        del reg_loss2, shortcut, grouped_points_nn, grouped_points, center_points, points
        if indices is None:
            return features, None, None, None, reg_loss, self.indices
        return features, None, None, None, reg_loss


class res_GAT_up(nn.Module):
    def __init__(self, k, n_cout, n_blocks, bn_decay=None, indices=None, up_ratio=2, use_bn=False, multi_gpus=False,
                 device=None, base_inp=128, base_oup=128, up_scales=np.arange(1.1, 9.1, 0.1), training=True,
                 drop_last_conv=False, metamask=False, bachnum=8):
        super(res_GAT_up, self).__init__()
        self.k = k
        self.bachnum = bachnum
        self.n_blocks = n_blocks
        self.indices = indices
        self.up_ratio = up_ratio
        self.use_bn = use_bn
        self.multi_gpus = multi_gpus
        self.device = device
        self.drop_last_conv = drop_last_conv
        self.withmetamask = metamask
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_cout, n_cout, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(n_cout, n_cout, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.metablock = res_GAT_meta_block(n_cout, multi_gpus=self.multi_gpus, base_inp=base_inp, base_oup=base_oup,
                                            up_scales=up_scales, training=training, k=k)
        self.block_list = nn.ModuleList((res_GAT_block(n_cout) for i in range(self.n_blocks)))
        self.block_list1 = nn.ModuleList((nn.Linear(128, 3) for i in range(self.n_blocks)))
        if self.drop_last_conv:
            self.last_block = res_GAT_block(n_cout, need_conv=False)
        self.max_ratio = math.ceil(math.sqrt(up_scales[-1]))
        if round(up_scales[-1], 1) == 9.0:
            self.max_ratio = 3
        self.conv3 = nn.Conv2d(n_cout, 3 * self.max_ratio, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv4 = nn.Conv2d(n_cout, 3 * self.max_ratio, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, xyz, points, up_scale, mask, Nout=0, indices=None):
        shortcut = points
        features = points
        npoint = int(xyz.numel() / (3 * self.bachnum))
        if Nout == 0:
            Nout = int(xyz.shape[1] * up_scale)
        up_scale = up_scale - (1e-6)
        features = self.relu(features)
        if indices is None:
            _, grouped_points, self.indices = group(xyz.detach(), features, self.k)
        else:
            self.indices = indices.detach()
            grouped_points = utils.grouping_operation(features, self.indices)
        center_points = torch.unsqueeze(features, 3)
        features = self.conv1(center_points)
        del center_points
        grouped_points_nn = self.conv2(grouped_points)
        del grouped_points
        features = torch.cat([features, grouped_points_nn], 3)
        features = torch.mean(features, 3, keepdim=False) + shortcut
        del grouped_points_nn

        features, _, _, _, reg_loss = self.metablock(features, up_scale, self.indices.detach())

        for i, block in enumerate(self.block_list):

            if self.drop_last_conv:
                features, _, _, _ = block(features, self.indices.detach())
            else:
                if i < self.n_blocks - 1:
                    features, _, _, _ = block(features, self.indices.detach())
                else:
                    features, center_points_, grouped_points_, grouped_points_nn_ = block(features,
                                                                                          self.indices.detach())

        if self.drop_last_conv:
            _, center_points_, grouped_points_, _ = self.last_block(features, self.indices.detach())
        points_xyz = self.conv3(center_points_)
        grouped_points_xyz = self.conv4(grouped_points_)
        del center_points_, grouped_points_
        new_xyz = torch.mean(torch.cat([points_xyz, grouped_points_xyz], 3), 3,
                             keepdim=True).transpose(2, 1).contiguous()
        del points_xyz, grouped_points_xyz
        new_xyz = new_xyz.view(new_xyz.shape[0], xyz.shape[1], self.max_ratio, 3).contiguous()

        new_xyz = (new_xyz + torch.unsqueeze(xyz, 2)).view(new_xyz.shape[0], -1, 3).contiguous()
        if mask is None:
            new_pred = new_xyz.transpose(1, 2).contiguous()
            new_xyz = utils.gather_operation(new_pred, utils.furthest_point_sample(new_xyz, Nout).detach()).transpose(1,
                                                                                                                      2).contiguous()
            del new_pred
        return new_xyz, features + shortcut, reg_loss


def get_uniform_loss(pcd, percentages=[0.004, 0.008, 0.010, 0.012, 0.016], radius=1.0):
    N = pcd.shape[1]
    npoint = int(N * 0.05)
    loss = torch.zeros([1], requires_grad=True).cuda(non_blocking=True)
    for p in percentages:
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) * p / nsample
        new_xyz = pcd.transpose(1, 2).contiguous()
        new_xyz = utils.gather_operation(new_xyz, utils.furthest_point_sample(pcd, npoint).detach())
        new_xyz = new_xyz.transpose(1, 2).contiguous()
        idx = utils.query_ball_point(r, nsample, new_xyz, pcd)
        del new_xyz
        expect_len = math.sqrt(2 * disk_area / 1.732)
        grouped_pcd = utils.grouping_operation(pcd.transpose(1, 2).contiguous(), idx.detach()).permute(0, 2, 3,
                                                                                                       1).contiguous()
        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), 0)
        uniform_dis, _ = knn_point1(2, grouped_pcd, grouped_pcd)
        uniform_dis = -uniform_dis[:, :, 1:]
        uniform_dis = torch.sqrt(torch.abs(uniform_dis + (1e-8)))
        uniform_dis = torch.mean(uniform_dis, -1)
        uniform_dis = (uniform_dis - expect_len) ** 2 / (expect_len + 1e-8)
        uniform_dis = uniform_dis.view(-1).contiguous()
        loss += torch.mean(uniform_dis, 0) * math.pow(p * 100, 2)
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
        del uniform_dis, grouped_pcd, expect_len
    return loss / len(percentages)


from model.loss_util import get_uniform_loss2, get_repulsion_loss


class GenModel(nn.Module):
    def __init__(self, bn_decay=None, up_ratio=4, use_bn=True, use_ibn=False, use_normal=False, multi_gpus=False,
                 device=None, training=True, bachnum=8):
        super(GenModel, self).__init__()
        self.use_bn = use_bn
        self.use_normal = use_normal
        self.bn_decay = bn_decay
        self.device = device
        self.bachnum = bachnum
        self.multi_gpus = multi_gpus
        self.training = training
        self.up_ratios = np.arange(1.1, 16.1, 0.1)
        self.base_inp, self.base_middle_inp, self.base_oup = 128, 128, 128
        self.pointcnn = pointcnn(8, 128, 9, bn_decay=bn_decay)
        self.res_GAT_up1 = res_GAT_up(8, 128, 2, bn_decay=bn_decay, up_ratio=2, use_bn=self.use_bn,
                                      multi_gpus=multi_gpus, device=device, base_inp=self.base_inp,
                                      base_oup=self.base_middle_inp, up_scales=self.up_ratios,
                                      training=training, bachnum=bachnum)  # changed (8,128,9)
        self.res_GAT_up2 = res_GAT_up(8, 128, 2, bn_decay=bn_decay, up_ratio=2, use_bn=self.use_bn,
                                      multi_gpus=multi_gpus, device=device, base_inp=self.base_inp,
                                      base_oup=self.base_middle_inp, up_scales=self.up_ratios, training=training,
                                      drop_last_conv=True, bachnum=bachnum)  # changed (8,128,10)
        self.max_ratio = math.ceil(math.sqrt(self.up_ratios[-1]))
        if round(self.up_ratios[-1], 1) == 9.0:
            self.max_ratio = 3

    def input_matrix_wpn(self, inN, scale, add_scale=True, outN=0):
        scale = round(scale, 3)
        mask = torch.zeros([inN, self.max_ratio, 1], dtype=torch.bool, requires_grad=False)
        if outN == 0:
            outN = int(inN * scale)
        int_project_coord = torch.floor(torch.arange(0, outN, 1).float().mul(1.0 / (scale))).int()
        flag = 0
        number = 0
        for i in range(outN):
            if int_project_coord[i] == number:
                try:
                    mask[int_project_coord[i], flag, 0] = 1
                except:
                    pass
                flag += 1
            else:
                try:
                    mask[int_project_coord[i], 0, 0] = 1
                except:
                    pass
                number += 1
                flag = 1
        del number, flag, int_project_coord
        mask_mat = mask.view(-1, self.max_ratio * inN, 1)
        del mask
        Nnow = mask_mat.sum()
        if Nnow < outN:
            for cnt in range(outN - Nnow):
                for i in range(mask_mat.shape[1]):
                    if mask_mat[0, i, :] == False:
                        mask_mat[:, i, :] = True
                        break
        if torch.__version__ == '1.3.0':
            if self.multi_gpus:
                mask_mat = mask_mat.type(torch.cuda.BoolTensor)
            else:
                mask_mat = mask_mat.cuda()
        else:
            if self.multi_gpus:
                mask_mat = mask_mat.byte().type(torch.cuda.BoolTensor)
            else:
                mask_mat = mask_mat.cuda().byte()
        return mask_mat, outN

    def forward(self, point_cloud, WD=None, pointclouds_gt=None, this_scale=4):
        xyz = point_cloud[:, :, 0:3]
        if self.use_normal:
            points = point_cloud[:, :, 3:]
        else:
            points = self.pointcnn(xyz, use_bn=self.use_bn)
        inN = xyz.shape[1]
        first_up_scale = math.sqrt(this_scale) + (1e-06)
        outN1 = int(inN * round(first_up_scale, 3))
        try:
            outN2 = pointclouds_gt.shape[1]
        except:
            outN2 = int(inN * this_scale)
        second_up_scale = (float(outN2) / float(outN1)) + (1e-06)

        new_xyz, points, reg_loss = self.res_GAT_up1(xyz, points, first_up_scale, None, Nout=outN1)
        del outN1, first_up_scale
        _, idx = knn_point1(8, xyz.detach(), new_xyz.detach())
        points = utils.grouping_operation(points, idx.detach())
        del idx
        points = torch.mean(points, 3, keepdim=False)

        new_xyz, _, reg_loss2 = self.res_GAT_up2(new_xyz, points, second_up_scale, None, Nout=outN2)
        del outN2, second_up_scale, points
        if self.training:
            uniform_loss = get_uniform_loss(new_xyz)
            repulsion_loss = get_repulsion_loss(new_xyz, device=self.device)
        else:
            uniform_loss, repulsion_loss = 0, 0
        reg_loss += reg_loss2
        del reg_loss2
        if WD is not None:
            if pointclouds_gt.shape[1] != new_xyz.shape[1]:
                print(pointclouds_gt.shape[1], new_xyz.shape[1])
            dist2 = WD(new_xyz, pointclouds_gt.detach())
            return new_xyz, None, dist2, reg_loss, uniform_loss, repulsion_loss, None
        return new_xyz, None, reg_loss, uniform_loss, repulsion_loss


if __name__ == '__main__':

    device = 'cuda:0'
    xyz = torch.rand(8, 250, 3)
    xyz = xyz.to(device)
    a = GenModel().to(device)
    y = a(xyz)
