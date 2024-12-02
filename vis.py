import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import os
import tqdm
import numpy as np
import argparse
import time
import copy
from dataset import GraspDataset
from model import backbone_pointnet2
import yaml
import loss_utils
from utils import scene_utils, pc_utils
import trimesh
import random

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CUR_PATH, 'config/base_config.yaml'), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser('Train DLR Grasp')
parser.add_argument('--batchsize', type=int, default=cfg['train']['batchsize'], help='input batch size')
parser.add_argument('--workers', type=int, default=cfg['train']['workers'], help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=cfg['train']['epoches'], help='number of epochs for training')
parser.add_argument('--gpu', type=str, default=cfg['train']['gpu'], help='specify gpu device')
parser.add_argument('--learning_rate', type=float, default=cfg['train']['learning_rate'],
                    help='learning rate for training')
parser.add_argument('--optimizer', type=str, default=cfg['train']['optimizer'], help='type of optimizer')
parser.add_argument('--theme', type=str, default=cfg['train']['theme'], help='type of train')
parser.add_argument('--model_path', type=str, default=cfg['eval']['model_path'], help='type of train')
FLAGS = parser.parse_args()


# for test
def vis_groundtruth(dataloader):
    taxonomy = ['Parallel_Extension', 'Pen_Pinch', 'Palmar_Pinch', 'Precision_Sphere', 'Large_Wrap']
    for i, (data, index) in enumerate(dataloader):
        img_id = index[0].numpy()
        point = copy.deepcopy(data['point'])[0]
        gt_list = []
        for k in data.keys():
            if data[k].size(-1) == 27:  # graspable 1, depth 1,quat 4 ,metric 1 ,joint 20
                gt_list.append(data[k])
        for i, gt in enumerate(gt_list):
            scene = trimesh.Scene()
            scene_mesh, _, _ = scene_utils.load_scene(img_id)
            scene.add_geometry(scene_mesh)
            pc = trimesh.PointCloud(point, colors=cfg['color']['pointcloud'])
            scene.add_geometry(pc)
            tax = taxonomy[i]
            graspable, pose_label, joint_label = gt[0, :, 0], gt[0, :, 1:5], gt[0, :, 7:]
            gp, pose, joint = point[graspable == 1], pose_label[graspable == 1], joint_label[graspable == 1]
            depth, azimuth, elevation, grasp_angle = pose[:, 0], pose[:, 1], pose[:, 2] + 90, pose[:, 3]
            approach_vector = loss_utils.angle_to_vector(azimuth.double(), elevation.double()).transpose(0,
                                                                                                         1)  # vector B*N, 3
            closing_vector = loss_utils.grasp_angle_to_vector(grasp_angle.double()).transpose(0, 1)
            R = loss_utils.rotation_from_vector(approach_vector, closing_vector)
            pos = gp + (approach_vector * (depth[:, np.newaxis] + 20.) / 100.)
            pos, R, joint = pos.numpy(), R.numpy(), joint.numpy()
            out_pos, out_quat, out_joint = scene_utils.decode_pred_new(pos, R, joint, tax)
            choice = np.random.choice(len(out_pos), 5, replace=True)
            out_pos, out_quat, out_joint = out_pos[choice], out_quat[choice], out_joint[choice]
            for p, q, j in zip(out_pos, out_quat, out_joint):
                # print(p,q,j)
                mat = trimesh.transformations.quaternion_matrix(q)
                hand_mesh = scene_utils.load_hand(p, q, j)
                scene.add_geometry(hand_mesh)
                scene.show()
                break


def vis_model(model, img_idx=0):
    model = model.eval()
    # Parallel_Extension：平行伸展
    # Pen_Pinch：笔式捏握
    # Palmar_Pinch：掌部捏握
    # Precision_Sphere：精确球握
    # Large_Wrap：大范围包握
    taxonomy = ['Parallel_Extension', 'Pen_Pinch', 'Palmar_Pinch', 'Precision_Sphere', 'Large_Wrap']
    # point [307200, 3] ; sem [307200,]的int

    # ------------------------------------------------------------------------------------------------------------------
    point, sem = scene_utils.load_scene_pointcloud(img_idx, use_base_coordinate=cfg['use_base_coordinate'],
                                                   split='test')
    # ------------------------------------------------------------------------------------------------------------------
    # data = np.load('../debug_scene01_data_for_hgcnet/debug_scene01.npz', allow_pickle=True)
    # point = data['pcd']
    # sem = data['seg'][:,0]
    # # 获取 num_points 的目标值
    # num_points = cfg['dataset']['num_points']
    # # 获取当前点的数量 n
    # n = point.shape[0]
    # if n < num_points:
    #     # 计算需要补全的数量
    #     num_to_pad = num_points - n
    #     # 用 [0, 0, 0] 补全 point
    #     padding_points = np.zeros((num_to_pad, 3))
    #     point_padded = np.vstack((point, padding_points))
    #     # 用 0 补全 sem
    #     padding_sem = np.zeros(num_to_pad, dtype=sem.dtype)
    #     sem_padded = np.concatenate((sem, padding_sem))
    # else:
    #     # 如果不需要补全，则原样返回
    #     point_padded = point
    #     sem_padded = sem
    # point, sem = point_padded, sem_padded

    # ------------------------------------------------------------------------------------------------------------------
    # import open3d as o3d
    # pcd = o3d.io.read_point_cloud("/home/kb/Downloads/testpcd.ply")
    # point = np.array(pcd.points)
    # sem = np.ones([point.shape[0]])

    # ==================================================================================================================
    # import open3d as o3d
    # cs = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point)
    # pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd, cs])
    # ==================================================================================================================
    # 加载场景的点云，并计算中心化后的归一化点云。
    center = np.mean(point, axis=0)
    norm_point = point - center
    crop_point, crop_index = pc_utils.crop_point(point)
    choice = np.random.choice(len(crop_point), cfg['dataset']['num_points'], replace=False)
    # point:[40000,3]; sem:[40000,]; norm_point:[40000,3]，都是ndarray
    point = point[crop_index][choice]
    sem = sem[crop_index][choice]
    norm_point = norm_point[crop_index][choice]

    # point_cloud = trimesh.PointCloud(point)
    # point_cloud.show()

    bat_point = copy.deepcopy(point)
    bat_sem = copy.deepcopy(sem)
    # for k in data.keys():
    #     data[k] = data[k].cuda().float()
    point = torch.tensor([point]).cuda().float()
    norm_point = torch.tensor([norm_point]).cuda().float()

    # !!!!!!!!!!!!!! 着重看这行 !!!!!!!!!!!!!!
    # bat_pred_graspable:[1,40000,2,5]
    # bat_pred_pose:     [1,40000,64,5]
    # bat_pred_joint:    [1,40000,20,5]
    bat_pred_graspable, bat_pred_pose, bat_pred_joint = \
        model(point, norm_point.transpose(1, 2))
    # gp的意思是graspable，即是否可抓；point:[40000,3];sem:[40000,],数值是0123这样;gp:[40000,2,5];pose:[40000,64,5];joint:[40000,20,5]
    point, sem, gp, pose, joint = \
        point[0].cpu(), bat_sem, bat_pred_graspable[0].cpu(), bat_pred_pose[0].cpu(), bat_pred_joint[0].cpu()

    # 打印 point 的最大值和最小值
    print(f"point: max={point.max()}, min={point.min()}")
    # 打印 sem 的最大值和最小值
    print(f"sem: max={sem.max()}, min={sem.min()}")
    # 打印 gp 的最大值和最小值
    print(f"gp: max={gp.max()}, min={gp.min()}")
    # 打印 pose 的最大值和最小值
    print(f"pose: max={pose.max()}, min={pose.min()}")
    # 打印 joint 的最大值和最小值
    print(f"joint: max={joint.max()}, min={joint.min()}")
    """
    point: max=0.49997562170028687,  min=-0.4999968111515045
    sem:   max=10.0,                 min=0.0
    gp:    max=14.096048355102539,   min=-13.947809219360352
    pose:  max=15.211660385131836,   min=-28.149578094482422
    joint: max=1.9571352005004883,   min=-0.38067761063575745
    """
    hand_meshes = []
    for t in range(gp.size(-1)):  # for each taxonomy
        tax = taxonomy[t]  # 当前抓取类型的标签名称
        scene = trimesh.Scene()
        scene_mesh, _, _ = scene_utils.load_scene(img_idx, split='test')
        # scene.add_geometry(scene_mesh)
        scene = scene_utils.add_scene_cloud(scene, bat_point)  # 加载点云
        # scene.show()
        # exit()
        tax_gp, tax_pose, tax_joint = gp[:, :, t], pose[:, :, t], joint[:, :, t]
        # print(tax_gp.shape,tax_pose.shape,tax_joint.shape)
        if cfg['train']['use_bin_loss']:  # True
            # point:, tax_gp, tax_pose, tax_joint
            out_gp, out_pos, out_R, out_joint, out_score = loss_utils.decode_pred(point, tax_gp, tax_pose,
                                                                                  tax_joint) # out_gp: 抓取bool分类;out_pos: 抓取的三维位置;out_R: 抓取的旋转矩阵;out_joint: 手部关节状态;out_score: 抓取置信度分数
            out_pos, out_R, out_joint, out_score = out_pos.detach().cpu().numpy(), \
                out_R.detach().cpu().numpy(), \
                out_joint.detach().cpu().numpy(), \
                out_score.detach().cpu().numpy()
        else:
            score = F.softmax(tax_gp, dim=1)[:, 1].detach().cpu().numpy()
            tax_gp, tax_pose, tax_joint = tax_gp.detach().cpu().numpy(), tax_pose.detach().cpu().numpy(), tax_joint.detach().cpu().numpy()
            depth, quat = tax_pose[:, 0], tax_pose[:, 1:]
            out_gp = np.argmax(tax_gp, 1)

            mat = trimesh.transformations.quaternion_matrix(quat)
            out_R = mat[:, :3, :3]
            approach = mat[:, :3, 2]
            offset = (depth / 100.0 * (approach.T)).T
            print(offset)
            out_pos = point + offset

            out_pos, out_R, out_joint, out_score = out_pos[out_gp == 1], out_R[out_gp == 1], tax_joint[out_gp == 1], \
            score[out_gp == 1]

        good_points = bat_point[out_gp == 1]
        # bad_points = point[out_gp == 0]
        scene = scene_utils.add_point_cloud(scene, good_points, color=cfg['color'][tax])
        scene.show()
        # exit()
        # scene = scene_utils.add_point_cloud(scene,bad_points,color = cfg['color']['bad_point'])
        for c in np.unique(sem):
            print(c)
            if c > 0.1:
                ins_pos, ins_R, ins_joint, ins_score = out_pos[sem[out_gp == 1] == c], \
                    out_R[sem[out_gp == 1] == c], \
                    out_joint[sem[out_gp == 1] == c], \
                    out_score[sem[out_gp == 1] == c]
                if len(ins_pos) > 0:
                    # ins_pos,ins_R,ins_joint = grasp_utils.grasp_nms(ins_pos,ins_R,ins_joint,ins_score)

                    ins_pos, ins_quat, ins_joint, mask = scene_utils.decode_pred_new(ins_pos, ins_R, ins_joint, tax)
                    # scene = scene_utils.add_point_cloud(scene,point[sem==c],color = [c*20,0,0])
                    for i, (p, q, j) in enumerate(zip(ins_pos, ins_quat, ins_joint)):
                        hand_mesh = scene_utils.load_hand(p, q, j, color=cfg['color'][taxonomy[t]])
                        hand_meshes.append(hand_mesh)
                        # hand_mesh = scene_utils.load_init_hand(p, q, init_hand, color=cfg['color'][taxonomy[t]])
                        scene.add_geometry(hand_mesh)
                        break
        scene.show()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print('Using GPUs ' + os.environ["CUDA_VISIBLE_DEVICES"])
    # dataset_path = os.path.join(CUR_PATH, '../data/point_grasp_data')
    # train_data = GraspDataset(dataset_path,split='train')
    # train_dataloader = torch.utils.data.DataLoader(train_data,
    #                                                batch_size = FLAGS.batchsize,
    #                                                shuffle=True,
    #                                                num_workers = FLAGS.workers)
    # test_data = GraspDataset(dataset_path,split='test')
    # test_dataloader = torch.utils.data.DataLoader(test_data,
    #                                            batch_size = FLAGS.batchsize,
    #                                            shuffle=True,
    #                                            num_workers = FLAGS.workers)
    model = backbone_pointnet2().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.load(os.path.join(f"{cfg['eval']['model_path']}/model_{str(cfg['eval']['epoch']).zfill(3)}.pth")))
    vis_model(model, 0)
    # vis_groundtruth(train_dataloader)

    # scene = trimesh.Scene()
    # scene_mesh, gt_objs, transform_list = scene_utils.load_scene(0, use_base_coordinate=cfg['use_base_coordinate'],
    #                                                              use_simplified_model=True, split='test')
    # scene.add_geometry(scene_mesh)
    # scene.show()

    # point, sem = scene_utils.load_scene_pointcloud(0, use_base_coordinate=cfg['use_base_coordinate'], split='test')
    # point_cloud = trimesh.PointCloud(sem)
    # point_cloud.show()
