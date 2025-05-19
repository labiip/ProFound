import argparse
import os
import numpy as np
import open3d as o3d
import cv2
import sys
import open3d as o3d
from tqdm import tqdm  
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from utils.nomarlization import *
from datasets.io import IO
from datasets.data_transforms import Compose
from utils.loss_utils import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def get_args():
    parser = argparse.ArgumentParser()
    # please modify here
    #------------------------------------------------------------------------------------
    parser.add_argument('--model_config', default='./cfgs/FID_Comp3D_models/ProFound.yaml',help = 'yaml config file')
    parser.add_argument('--model_checkpoint', default= ' ',help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='data/FID_Comp3D/test/partial/0/', help='Pc root')
    parser.add_argument('--out_mertic_log', type=str, default='vis/output_log.txt')
    parser.add_argument('--out_pc_root',type=str,default='vis/',help='root of the output pc file. ''Default not saving the visualization images.')
    #------------------------------------------------------------------------------------
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument('--save_vis_img',action='store_true',default=True,help='whether to save img of complete point cloud') 
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, gt_path, args, config, root=None):

    # read single point cloud
    pc_ndarray = IO.get(pc_path).astype(np.float32)
    gt_ndarray = IO.get(gt_path).astype(np.float32)

    # transform it according to the model 
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # normalize it to fit the model on ShapeNet-55/34
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    pred_points = torch.from_numpy(dense_points.reshape(1, -1, 3))
    pred_points = pred_points.cuda()
    gt_points = torch.from_numpy(gt_ndarray.reshape(1, -1, 3))
    gt_points = gt_points.cuda()

    batch_size, n_x, _ = pred_points.shape
    batch_size, n_gt, _ = gt_points.shape
    assert pred_points.shape[0] == gt_points.shape[0]

    output_file = args.out_mertic_log
    if not Path(output_file).exists():
        # print(f"文件 {output_file} 不存在，正在创建新文件...")
        with open(output_file, "w") as file:
            pass  # 创建一个空文件

    # 构造要写入的字符串
    log_entry = f"Input: {pc_path} --> metric: {chamfer_sqrt(pred_points, gt_points)* 1000}\n"  

    # 追加写入文件
    with open(output_file, "a") as file:  # 使用 'a' 模式追加写入
        file.write(log_entry)  # 写入一个字符串

    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # denormalize it to adapt for the original input
        dense_points = dense_points * m
        dense_points = dense_points + centroid

    # 保存结果
    if args.out_pc_root != '':

        # 创建输出目录，保持与输入文件相同的目录结构
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 保存点云数据为 .npy 文件
        # np.save(os.path.join(target_path, 'fine.npy'), dense_points)

        # 保存点云数据为 .pcd 文件
        o3d_data = o3d.geometry.PointCloud()
        o3d_data.points = o3d.utility.Vector3dVector(dense_points)
        output_filename = os.path.join(target_path + '_fine.pcd')
        o3d.io.write_point_cloud(output_filename, o3d_data, write_ascii=True)

        # # 保存可视化图像（如果启用）
        # if args.save_vis_img:
        #     input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
        #     dense_img = misc.get_ptcloud_img(dense_points)
        #     cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
        #     cv2.imwrite(os.path.join(target_path, 'fine.jpg'), dense_img)
    
    return

def find_pcd_files(directory):
    """
    递归查找目录下所有 .pcd 文件
    :param directory: 根目录
    :return: 所有 .pcd 文件的路径列表
    """
    pcd_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pcd"):
                pcd_files.append(os.path.join(root, file))
    return pcd_files

def find_gt_files(pcd_files):
    gt_files = []
    for pcd_file in pcd_files:
        # 将路径中的 'partial' 替换为 'complete'
        pcd_file = pcd_file.replace('partial', 'complete')
        # 将路径按 '/' 分割，去掉最后一个部分（文件名）
        parts = pcd_file.split('/')
        # 去掉最后一个部分后重新拼接成路径
        gt_file = '/'.join(parts[:-1]) + '.pcd'
        gt_files.append(gt_file)
    return gt_files

def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    # 批量处理点云数据
    if args.pc_root != '':
        # 递归查找所有 .pcd 文件
        pcd_files = find_pcd_files(args.pc_root)
        gt_files = find_gt_files(pcd_files)

        # for i in range(len(pcd_files)):
        #     print("index:",i+1 ,"inut:",pcd_files[i], "gt:",gt_files[i])

        if not pcd_files:
            print(f"No .pcd files found in {args.pc_root}!")
            return

        # 使用 tqdm 显示进度条
        for pcd_file, gt_file in tqdm(zip(pcd_files, gt_files), desc="Processing point clouds"):
            try:
                # 调用 inference_single 处理单个文件
                inference_single(base_model, pcd_file, gt_file, args, config)
            except Exception as e:
                print(f"Error processing {pcd_file}: {e}")
    else:
        # 处理单个点云文件
        if not os.path.exists(args.pc):
            print(f"Point cloud file {args.pc} does not exist!")
            return
        try:
            inference_single(base_model, args.pc, args, config)
        except Exception as e:
            print(f"Error processing {args.pc}: {e}")

    print("Processing completed!")


if __name__ == '__main__':
    main()



#---------inference---------#
    # if args.pc_root != '':
    #     pc_file_list = os.listdir(args.pc_root)
    #     for pc_file in pc_file_list:
    #         inference_single(base_model, pc_file, args, config, root=args.pc_root)
    # else:
    #     inference_single(base_model, args.pc, args, config)

#---------normalize---------#
# # (n,c) -> (1,n,c)
    # pc_ndarray = pc_ndarray.reshape(1,2048,3)

    # Normalization
    # pc_ndarray = torch.from_numpy(pc_ndarray)
    
    # # pc_ndarray, centroid, furthest_distance=normalize_point_cloud(pc_ndarray)
    # # centroid = centroid.cuda()
    # # furthest_distance = furthest_distance.cuda()   

    # ret = list(ret)
    # num_list = len(ret)

    # pc_ndarray = pc_ndarray.numpy()
    # # (1,n,c) -> (n,c)
    # pc_ndarray = np.squeeze(pc_ndarray)
            
    # for i in range(num_list):
    #     ret[i] = ret[i] * furthest_distance + centroid
    # del centroid
    # del furthest_distance