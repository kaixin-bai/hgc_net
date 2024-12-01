import sys
import argparse
import os
import time
import pickle
import scipy.spatial
import torch
import math
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import functools
import yaml
with open('config/base_config.yaml', 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)

azimuth_scope = cfg['bin_loss']['azimuth_scope']
azimuth_bin_size = cfg['bin_loss']['azimuth_bin_size']
elevation_scope = cfg['bin_loss']['elevation_scope']
elevation_bin_size = cfg['bin_loss']['elevation_bin_size']
grasp_angle_scope = cfg['bin_loss']['grasp_angle_scope']
grasp_angle_bin_size = cfg['bin_loss']['grasp_angle_bin_size']
depth_scope = cfg['bin_loss']['depth_scope']
depth_bin_size = cfg['bin_loss']['depth_bin_size']

per_azimuth_bin_num = int(azimuth_scope / azimuth_bin_size)
per_elevation_bin_num = int(elevation_scope / elevation_bin_size)
per_depth_bin_num = int(depth_scope / depth_bin_size)
per_grasp_angle_bin_num = int(grasp_angle_scope / grasp_angle_bin_size)

def get_bin_reg_loss(pred_pose, gt_pose):
    loss_dict = {}
    loc_loss = 0
    start_offset = 0

    depth,azimuth_angle,elevation_angle,grasp_angle = gt_pose[:,0],gt_pose[:,1],gt_pose[:,2],gt_pose[:,3]

    per_depth_bin_num = int(depth_scope / depth_bin_size)
    depth_bin_l, depth_bin_r = start_offset, start_offset + per_depth_bin_num
    depth_res_l, depth_res_r = depth_bin_r, depth_bin_r + per_depth_bin_num
    start_offset = depth_res_r

    depth_shift = torch.clamp(depth, 0, depth_scope - 1e-4)
    depth_bin_label = (depth_shift / depth_bin_size).floor().long()
    #depth_res_label = depth_shift - (depth_bin_label.float() * depth_bin_size + depth_bin_size / 2)
    depth_res_label = depth_shift - (depth_bin_label.float() * depth_bin_size)
    depth_res_norm_label = depth_res_label / depth_bin_size

    depth_bin_onehot = torch.cuda.FloatTensor(depth_bin_label.size(0), per_depth_bin_num).zero_()
    depth_bin_onehot.scatter_(1, depth_bin_label.view(-1, 1).long(), 1)

    loss_depth_bin = F.cross_entropy(pred_pose[:, depth_bin_l: depth_bin_r], depth_bin_label)
    loss_depth_res = F.smooth_l1_loss((torch.sigmoid(pred_pose[:, depth_res_l: depth_res_r]) * depth_bin_onehot).sum(dim=1), depth_res_norm_label)
    loss_dict['depth_bin_loss'] = loss_depth_bin.item()
    loss_dict['depth_res_loss'] = loss_depth_res.item()
    loc_loss += loss_depth_bin + loss_depth_res

    azimuth_bin_l, azimuth_bin_r = start_offset, start_offset + per_azimuth_bin_num
    azimuth_res_l, azimuth_res_r = azimuth_bin_r, azimuth_bin_r + per_azimuth_bin_num
    start_offset = azimuth_res_r

    azimuth_shift = torch.clamp(azimuth_angle, 0, azimuth_scope - 1e-4)
    azimuth_bin_label = (azimuth_shift / azimuth_bin_size).floor().long()
    #azimuth_res_label = azimuth_shift - (azimuth_bin_label.float() * azimuth_bin_size + azimuth_bin_size / 2)
    azimuth_res_label = azimuth_shift - (azimuth_bin_label.float() * azimuth_bin_size)
    azimuth_res_norm_label = azimuth_res_label / azimuth_bin_size

    azimuth_bin_onehot = torch.cuda.FloatTensor(azimuth_bin_label.size(0), per_azimuth_bin_num).zero_()
    azimuth_bin_onehot.scatter_(1, azimuth_bin_label.view(-1, 1).long(), 1)

    loss_azimuth_bin = F.cross_entropy(pred_pose[:, azimuth_bin_l: azimuth_bin_r], azimuth_bin_label)
    loss_azimuth_res = F.smooth_l1_loss((torch.sigmoid(pred_pose[:, azimuth_res_l: azimuth_res_r]) * azimuth_bin_onehot).sum(dim=1), azimuth_res_norm_label)
    loss_dict['azimuth_bin_loss'] = loss_azimuth_bin.item()
    loss_dict['azimuth_res_loss'] = loss_azimuth_res.item()
    loc_loss += loss_azimuth_bin + loss_azimuth_res


    elevation_bin_l, elevation_bin_r = start_offset, start_offset + per_elevation_bin_num
    elevation_res_l, elevation_res_r = elevation_bin_r, elevation_bin_r + per_elevation_bin_num
    start_offset = elevation_res_r

    elevation_shift = torch.clamp(elevation_angle, 0, elevation_scope - 1e-4)
    elevation_bin_label = (elevation_shift / elevation_bin_size).floor().long()
    #elevation_res_label = elevation_shift - (elevation_bin_label.float() * elevation_bin_size + elevation_bin_size / 2)
    elevation_res_label = elevation_shift - (elevation_bin_label.float() * elevation_bin_size)
    elevation_res_norm_label = elevation_res_label / elevation_bin_size

    elevation_bin_onehot = torch.cuda.FloatTensor(elevation_bin_label.size(0), per_elevation_bin_num).zero_()
    elevation_bin_onehot.scatter_(1, elevation_bin_label.view(-1, 1).long(), 1)

    loss_elevation_bin = F.cross_entropy(pred_pose[:, elevation_bin_l: elevation_bin_r], elevation_bin_label)
    loss_elevation_res = F.smooth_l1_loss((torch.sigmoid(pred_pose[:, elevation_res_l: elevation_res_r]) * elevation_bin_onehot).sum(dim=1), elevation_res_norm_label)
    loss_dict['elevation_bin_loss'] = loss_elevation_bin.item()
    loss_dict['elevation_res_loss'] = loss_elevation_res.item()

    loc_loss += loss_elevation_bin + loss_elevation_res

    grasp_angle_bin_l, grasp_angle_bin_r = start_offset, start_offset + per_grasp_angle_bin_num
    grasp_angle_res_l, grasp_angle_res_r = grasp_angle_bin_r, grasp_angle_bin_r + per_grasp_angle_bin_num
    start_offset = grasp_angle_res_r
    # print('start_offset',start_offset)
    grasp_angle_shift = torch.clamp(grasp_angle, 0, grasp_angle_scope - 1e-4)
    grasp_angle_bin_label = (grasp_angle_shift / grasp_angle_bin_size).floor().long()
    #grasp_angle_res_label = grasp_angle_shift - (grasp_angle_bin_label.float() * grasp_angle_bin_size + grasp_angle_bin_size / 2)
    grasp_angle_res_label = grasp_angle_shift - (grasp_angle_bin_label.float() * grasp_angle_bin_size)
    grasp_angle_res_norm_label = grasp_angle_res_label / grasp_angle_bin_size

    grasp_angle_bin_onehot = torch.cuda.FloatTensor(grasp_angle_bin_label.size(0), per_grasp_angle_bin_num).zero_()
    grasp_angle_bin_onehot.scatter_(1, grasp_angle_bin_label.view(-1, 1).long(), 1)

    loss_grasp_angle_bin = F.cross_entropy(pred_pose[:, grasp_angle_bin_l: grasp_angle_bin_r], grasp_angle_bin_label)
    loss_grasp_angle_res = F.smooth_l1_loss((torch.sigmoid(pred_pose[:, grasp_angle_res_l: grasp_angle_res_r]) * grasp_angle_bin_onehot).sum(dim=1), grasp_angle_res_norm_label)
    loss_dict['grasp_angle_bin_loss'] = loss_grasp_angle_bin.item()
    loss_dict['grasp_angle_res_loss'] = loss_grasp_angle_res.item()
    loc_loss += loss_grasp_angle_bin + loss_grasp_angle_res

    loss_dict['loss_loc'] = loc_loss

    return loss_dict,loc_loss

def angle_to_vector(azimuth_angle, elevation_angle):
    """
    将球坐标系中的方位角和仰角转换为三维单位向量
    方位角决定 x 和 y 分量，仰角决定 z 分量
    """
    device = azimuth_angle.device
    x_ = torch.cos(azimuth_angle/180*math.pi)
    y_ = torch.sin(azimuth_angle/180*math.pi)
    z_ = -torch.tan((elevation_angle)/180 * math.pi) * torch.sqrt(x_**2 + y_**2)
    approach_vector_ = torch.stack((x_, y_, z_), axis=0)
    approach_vector = torch.div(approach_vector_, torch.norm(approach_vector_, dim=0))  # approach_vector:(3,n)
    return approach_vector

def grasp_angle_to_vector(grasp_angle):
    device = grasp_angle.device
    x_ = torch.cos(grasp_angle/180*math.pi)
    y_ = torch.sin(grasp_angle/180*math.pi)
    z_ = torch.zeros(grasp_angle.shape[0]).to(device)
    x_, y_, z_ = x_.double(), y_.double(), z_.double()
    closing_vector = torch.stack((x_, y_, z_), axis=0)

    return closing_vector

def rotation_from_vector(approach_vector, closing_vector):
    #TODO: if element in approach_vector[:, 2] = 0 cause error !
    temp = -(approach_vector[:, 0] * closing_vector[:, 0] + approach_vector[:, 1] * closing_vector[:, 1]) / approach_vector[:, 2]
    closing_vector[:, 2] = temp
    closing_vector = torch.div(closing_vector.transpose(0, 1), torch.norm(closing_vector, dim=1)).transpose(0, 1)
    z_axis = torch.cross(approach_vector.float(), closing_vector.float(), dim = 1)
    R = torch.stack((-z_axis, closing_vector.float(), approach_vector.float()), dim=-1)
    return R


def decode_pred(point, pred_gp, pred_pose, pred_joint):
    """
    Args:
        point: [40000, 3]  点云
        pred_gp: [40000, 2]  每种抓取的分类
        pred_pose: [40000, 20]  一个手20个自由度的数值
        pred_joint: [40000, 64]  理论上这里应该是手腕的Pose吧，为什么是64个数？

        如论文所说，他们在数据生成阶段首先在物体上采样了512个点，然后在法向方向选择了几个不同的depth区间
    """
    # print(pred_gp.size())
    out_gp = torch.argmax(pred_gp, dim=1).bool()  # 找出每个点的抓取可能性最大值对应的索引。由于 pred_gp 有两个值（可抓取和不可抓取），这里返回的是可抓取（True）或不可抓取（False）。这里是为了得到5种抓取方式中的max索引，{[}Tensor:(40000,2)},是bool值，即True或者False
    score = F.softmax(pred_gp, dim=1)[:, 1]  # score:{Tensor:(40000,)}。计算每个点属于抓取类的概率分布
    score = score[out_gp]  # 只取属于抓取类的分数，用于衡量抓取质量

    if torch.sum(out_gp) <= 0:  # 如果没有点被预测为可抓取，函数直接返回。
        return

    # 根据抓取点筛选姿态和关节状态，筛选出预测为可抓取的点的姿态和关节状态，下面两行注释里只看后面的维度就行，前面的是动态变化的
    pred_pose = pred_pose[out_gp]  # pred_pose: [1640, 64]
    joint = pred_joint[out_gp]  # (1640,20)

    #  解析抓取深度
    # 定义起始偏移量 start_offset，用于标记 pred_pose 中每段预测数据的起始位置
    # 每解码一部分数据（例如深度、方位角等），需要根据其分箱数量和残差范围更新 start_offset，以正确解析下一部分数据
    start_offset = 0

    """
    这个部分的设计和自动驾驶的point cloud based 3d object detection一样
    把要预测的6d pose拆分为不同的bin，先通过分类来得到pose是哪个bin中的，然后在每个bin里做回归计算offset；分类这里给出的是int的类别值，而回归部分归一化为0-1的float
    以下depth的含义见V.Method => C.Wrist Pose Estimation，这个是approach depth
    """
    # depth_bin_l 和 depth_bin_r：标记深度分箱的索引范围，用于分类。仔细来说就是我们从左到右定义一些bin，当前是从0到0+8。per_depth_bin_num是深度分箱的数量
    # depth_bin_l, depth_bin_r: pred_pose 中深度分箱分类的起始和结束索引
    depth_bin_l, depth_bin_r = start_offset, start_offset + per_depth_bin_num  # depth_bin_l:0  depth_bin_r:8
    # depth_res_l 和 depth_res_r：标记深度回归的索引范围，用于回归。从8到8+8
    # depth_res_l, depth_res_r: pred_pose 中深度回归预测的起始和结束索引
    depth_res_l, depth_res_r = depth_bin_r, depth_bin_r + per_depth_bin_num  # depth_res_l: 8  depth_res_r: 16

    # start_offset: 更新后指向深度部分的结束位置，用于下一部分解码
    start_offset = depth_res_r  # start_offset: 8 ==> 16
    # 在深度分箱范围 [depth_bin_l: depth_bin_r] 内，找到概率最大的分箱索引 depth_bin，这里一组是8个分箱，这个就是depth的分类结果了
    depth_bin = torch.argmax(pred_pose[:, depth_bin_l: depth_bin_r], dim=1)  # pred_pose:[1804,64]; depth_bin: {Tensor:(1804,)}, depth_bin里是int数值，表示的是第几个分箱的概率是最大的
    # 使用 depth_bin 的索引，从深度残差范围 [depth_res_l: depth_res_r] 提取回归值 depth_res_norm；
    # torch.gather 的作用是从 pred_pose 的特定位置提取数据；depth_res_norm:
    # 每个点的深度回归值，归一化到 [0, 1]
    depth_res_norm = torch.gather(pred_pose[:, depth_res_l: depth_res_r], dim=1, index=depth_bin.unsqueeze(dim=1)).squeeze(dim=1)

    # 计算真实深度：分箱部分的深度：depth_bin.float() * depth_bin_size；分箱中心偏移量：depth_bin_size / 2；残差部分的深度：depth_res
    # depth_bin_size: 每个分箱的深度大小
    depth_res = depth_res_norm * depth_bin_size
    # depth: 抓取点的真实深度
    depth = depth_bin.float() * depth_bin_size + depth_bin_size / 2 + depth_res
    # azimuth方位角 和 elevation仰角；azimuth方位角是60度一个bin。360度分为6个bin，每个bin是60度，然后res是大概-1.0~1.0，然后再乘以60.0度
    azimuth_bin_l, azimuth_bin_r = start_offset, start_offset + per_azimuth_bin_num  # start_offset=16，per_azimuth_bin_num=6，azimuth_bin_l=16，azimuth_bin_r=22
    azimuth_res_l, azimuth_res_r = azimuth_bin_r, azimuth_bin_r + per_azimuth_bin_num # 22，22+6=28    azimuth_res_l:22    azimuth_res_r:28
    start_offset = azimuth_res_r  # 16 ==> 28
    azimuth_bin = torch.argmax(pred_pose[:, azimuth_bin_l: azimuth_bin_r], dim=1)  # azimuth_bin是一个Tensor{(1697,)},其中的数值是int表示的是分类bin落在哪里。这里是从pred_pose所有点的16-22列取值
    azimuth_res_norm = torch.gather(pred_pose[:, azimuth_res_l: azimuth_res_r], dim=1, index=azimuth_bin.unsqueeze(dim=1)).squeeze(dim=1) # pred_pose所有行的22-28列，float值，正负都有
    azimuth_res = azimuth_res_norm * azimuth_bin_size  # azimuth_bin_size:60  看起来是把res转成'度'了，这里应该是大概-60 --> 60度左右
    azimuth_angle = azimuth_bin.float() * azimuth_bin_size + azimuth_bin_size / 2 + azimuth_res # 这里是通过bin和res把绝对的数值计算出来
    # 这一段就没什么好说的了，总共90度，6个bin，每个bin的res值是15度的范围，这里的取值范围是90-180度
    elevation_bin_l, elevation_bin_r = start_offset, start_offset + per_elevation_bin_num  # 28，28+6=34
    elevation_res_l, elevation_res_r = elevation_bin_r, elevation_bin_r + per_elevation_bin_num  # 34，34+6=40
    start_offset = elevation_res_r  # 28 -> 40
    elevation_bin = torch.argmax(pred_pose[:, elevation_bin_l: elevation_bin_r], dim=1)  # 这里是Tensor:(1697,)，数值是int，即0-某个值
    elevation_res_norm = torch.gather(pred_pose[:, elevation_res_l: elevation_res_r], dim=1, index=elevation_bin.unsqueeze(dim=1)).squeeze(dim=1)  # 1697个float值，大概是-1.0~1.0
    elevation_res = elevation_res_norm * elevation_bin_size  # elevation_bin_size=15；90 / 15 = 6
    elevation_angle = elevation_bin.float() * elevation_bin_size + elevation_bin_size / 2 + elevation_res #
    elevation_angle = elevation_angle + 90  # 这里的加90，意思是elevation_angle的取值范围是90-180度

    approach_vector = angle_to_vector(azimuth_angle, elevation_angle).transpose(0, 1) #vector B*N, 3
    #print(approach_vector[0])
    #
    grasp_angle_bin_l, grasp_angle_bin_r = start_offset, start_offset + per_grasp_angle_bin_num  # 40，40+12=52
    grasp_angle_res_l, grasp_angle_res_r = grasp_angle_bin_r, grasp_angle_bin_r + per_grasp_angle_bin_num # 52，52+12=64
    start_offset = grasp_angle_res_r  # 40 -> 64
    grasp_angle_bin = torch.argmax(pred_pose[:, grasp_angle_bin_l: grasp_angle_bin_r], dim=1)
    grasp_angle_res_norm = torch.gather(pred_pose[:, grasp_angle_res_l: grasp_angle_res_r], dim=1, index=grasp_angle_bin.unsqueeze(dim=1)).squeeze(dim=1)
    grasp_angle_res = grasp_angle_res_norm * grasp_angle_bin_size  # grasp_angle_bin_size:30，即grasp angle的一个bin是30度
    grasp_angle = grasp_angle_bin.float() * grasp_angle_bin_size + grasp_angle_bin_size / 2 + grasp_angle_res

    closing_vector = grasp_angle_to_vector(grasp_angle).transpose(0, 1)
    R = rotation_from_vector(approach_vector, closing_vector)  # 手腕的旋转
    #print(R[0])
    approach = R[:,:3,2] # the last column
    gp = point[out_gp]
    # print(approach.shape,gp.shape,depth.shape)
    pos = gp + (approach * (depth[:, np.newaxis] + 20.) / 100.)  # tensor,(n,3),这里应该是手腕的xyz
    # 抓取可行性out_gp、抓取位置pos、旋转矩阵R、关节状态joint,以及抓取评分score
    return out_gp, pos, R, joint, score
