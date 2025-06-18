'''
视觉导航数据集处理模块

本模块提供了处理视觉导航任务所需的数据集功能，包括：
1. 加载和处理指令文本
2. 处理视觉特征（场景视图和物体特征）
3. 构建导航轨迹
4. 生成训练和评估所需的输入数据

主要组件：
- 深度特征数据库 (DepthFeaturesDB)
- 语义特征数据库 (SemanticFeaturesDB)
- 视觉导航数据集基类 (ReverieTextPathData)
- R2R数据集处理类 (R2RTextPathData)
- SOON数据集处理类 (SoonTextPathData)
'''
import math
import os
import random
from collections import defaultdict
import jsonlines
import numpy as np
import torch

from .common import calculate_vp_rel_pos_fts
from .common import get_angle_fts, get_view_rel_angles
from .common import load_nav_graphs
from .common import softmax

# 归一化参数
MAX_DIST = 30  # 距离归一化最大值（米）
MAX_STEP = 10  # 评估阶段步数归一化最大值
TRAIN_MAX_STEP = 20  # 训练阶段最大步数限制

import json
import h5py


def load_viewpoint_ids(connectivity_dir):
    """
    从连接图数据中加载所有可用的视点ID
    
    参数:
        connectivity_dir (str): 包含场景连接图数据的目录路径
        
    返回:
        list: 包含(场景ID, 视点ID)元组的列表，每个元组代表一个可用的观察点
        
    说明:
        - 首先从scans.txt文件中读取所有场景ID
        - 然后对每个场景加载其连接图(connectivity.json)
        - 仅包含标记为included的视点
    """
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids


# 数据结构相关常量
TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']  # TSV文件字段名
VIEWPOINT_SIZE = 36  # 每个视点的离散视角数量

# 图像和地图尺寸参数
WIDTH = 128  # 图像宽度（像素）
HEIGHT = 128  # 图像高度（像素）
VFOV = 60  # 垂直视场角（度）
GLOBAL_WIDTH = 16  # 全局地图宽度（网格单元）
GLOBAL_HEIGHT = 16  # 全局地图高度（网格单元）

# 导航评估参数
ERROR_MARGIN = 3.0  # 到达目标的误差容限（米）



class DepthFeaturesDB(object):
    """
    深度特征数据库类
    
    用于加载和管理场景中各视点的深度特征。深度特征用于构建场景的3D结构理解，
    存储在HDF5格式的文件中。每个视点的深度特征以uint16类型存储，表示场景中
    物体的深度信息。
    """
    def __init__(self, img_ft_file):
        """
        初始化深度特征数据库
        
        参数:
            img_ft_file (str): HDF5格式的深度特征文件路径
        """
        self.img_ft_file = img_ft_file
        self._feature_store = h5py.File(self.img_ft_file, 'r')

    def get_image_feature(self, scan, viewpoint):
        """
        获取指定场景和视点的深度特征
        
        参数:
            scan (str): 场景ID
            viewpoint (str): 视点ID
            
        返回:
            np.ndarray: 深度特征数组，类型为uint16，表示场景中物体的深度值
        """
        key = '%s_%s' % (scan, viewpoint)
        ft = self._feature_store[key][...][:].astype(np.uint16)
        return ft


class SemanticFeaturesDB(object):
    """
    语义特征数据库类
    
    用于加载和管理场景中各视点的语义特征。语义特征用于理解场景中的物体和
    空间关系，存储在HDF5格式的文件中。每个视点的语义特征以float32类型存储，
    表示场景中物体的语义信息。
    """
    def __init__(self, img_ft_file):
        """
        初始化语义特征数据库
        
        参数:
            img_ft_file (str): HDF5格式的语义特征文件路径
        """
        self.img_ft_file = img_ft_file
        self._feature_store = h5py.File(self.img_ft_file, 'r')

    def get_image_feature(self, scan, viewpoint):
        """
        获取指定场景和视点的语义特征
        
        参数:
            scan (str): 场景ID
            viewpoint (str): 视点ID
            
        返回:
            np.ndarray: 语义特征数组，类型为float32，表示场景中物体的语义嵌入
        """
        key = '%s_%s' % (scan, viewpoint)
        ft = self._feature_store[key][...][:].astype(np.float32)
        return ft


class SingleTextDescriptionDB(object):
    """
    文本描述特征数据库类，用于加载和管理场景中各视点的文本描述嵌入向量。
    
    该类支持两种类型的文本描述：
    1. object: 描述视点中可见物体的文本嵌入
    2. spatial: 描述视点中空间关系的文本嵌入
    
    每个视点包含36个视角(view)，每个视角对应一个文本描述嵌入向量。
    """

    def __init__(self, pt_file, mode='object'):
        """
        初始化文本描述数据库
        
        参数:
            pt_file (str): 包含预训练文本嵌入的PyTorch文件路径
            mode (str): 文本描述类型，可选值为'object'或'spatial'
        """
        assert mode in ['object', 'spatial']
        self.mode = mode
        # 默认为每个视点的36个视角创建零向量(维度768)
        self.text_data = defaultdict(lambda: [torch.zeros(768)] * 36)

        if not os.path.exists(pt_file):
            print(f"[Warning] TextDescriptionDB: file not found at {pt_file}")
            return

        # 加载预训练的文本嵌入
        loaded = torch.load(pt_file)
        embeddings = loaded[f'{mode}_embeddings']  # 例如 object_embeddings
        meta_infos = loaded['meta_infos']

        # 将嵌入向量映射到对应的场景-视点-视角
        for info, emb in zip(meta_infos, embeddings):
            key = f"{info['scan_id']}_{info['viewpoint_id']}"
            idx = info['view_index']
            if self.text_data[key][idx] is None:
                self.text_data[key][idx] = emb

    def get_text_features(self, scan, viewpoint):
        """
        获取指定场景和视点的所有视角文本特征
        
        参数:
            scan (str): 场景ID
            viewpoint (str): 视点ID
            
        返回:
            List[torch.Tensor]: 包含36个文本嵌入向量的列表，每个向量维度为768
                               对应视点的36个不同视角的文本描述
        """
        key = f"{scan}_{viewpoint}"
        return self.text_data[key]  # List[Tensor[768]] (长度36)
    
def get_rel_position(depth_map, angle):
    """
    计算深度图中每个像素相对于观察点的相对位置坐标
    
    参数:
        depth_map (np.ndarray): 深度图数据，表示每个像素的深度值
        angle (float): 观察角度（弧度），用于坐标变换
        
    返回:
        tuple: (rel_x, rel_y) 两个数组，表示每个像素点相对于观察点的x和y坐标
        
    说明:
        该函数将深度图转换为相对坐标，考虑了观察角度的影响，用于构建全局地图
    """
    h, w = depth_map.shape  # 支持任意尺寸，比如 8x8
    depth_y = depth_map.astype(np.float32) / 4000.  # 深度值归一化

    # 构造水平方向偏移索引（范围从 -1 到 1）
    x_index = np.linspace(-1, 1, w, dtype=np.float32)
    x_offset = np.tile(x_index, (h, 1)) * math.tan(math.pi / 6)  # 假设水平视角为60°

    depth_x = depth_y * x_offset  # 计算x方向的深度投影

    # 根据观察角度进行坐标旋转变换
    rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
    rel_y = depth_y * math.cos(angle) - depth_x * math.sin(angle)

    return rel_x, rel_y


class ReverieTextPathData(object):
    """
    REVERIE数据集的路径数据处理基类
    
    该类负责加载和处理REVERIE数据集中的导航路径数据，包括：
    - 图像特征（场景视图）
    - 物体特征
    - 文本指令
    - 导航路径
    - 全局和局部地图信息
    """
    def __init__(
            self, anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=768, image_prob_size=1000, angle_feat_size=4,
            obj_feat_size=None, obj_prob_size=None, max_objects=20,
            max_txt_len=100, in_memory=True, is_train=False, act_visited_node=False,
            semantic_map_dir="../datasets/R2R/features"
    ):
        """
        初始化REVERIE路径数据处理器
        
        参数:
            anno_files (list): 包含数据集标注的文件路径列表
            img_ft_file (str): 图像特征文件路径
            obj_ft_file (str): 物体特征文件路径
            scanvp_cands_file (str): 视点候选文件路径
            connectivity_dir (str): 场景连接图数据目录
            image_feat_size (int): 图像特征维度，默认768
            image_prob_size (int): 图像概率特征维度，默认0
            angle_feat_size (int): 角度特征维度，默认4
            obj_feat_size (int): 物体特征维度
            obj_prob_size (int): 物体概率特征维度
            max_objects (int): 每个视点最大物体数量，默认20
            max_txt_len (int): 文本指令最大长度，默认100
            in_memory (bool): 是否将特征加载到内存，默认True
            is_train (bool): 是否为训练模式，默认False
            act_visited_node (bool): 是否只考虑已访问节点的动作，默认False
            semantic_map_dir (str): 语义地图特征目录
        """
        self.is_train = is_train
        self.img_ft_file = img_ft_file
        self.obj_ft_file = obj_ft_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in
                                    self.all_point_rel_angles]

        self.data = []
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                for item in f:
                    self.data.append(item)

        self.global_semantic = []
        self.global_spatial_text = []
        self.global_position_x = []
        self.global_position_y = []
        self.global_mask = []
        self.max_x = -10000
        self.min_x = 10000
        self.max_y = -10000
        self.min_y = 10000
        self.heading = 0
        self.global_map = None

        self.DepthDB = DepthFeaturesDB(os.path.join(semantic_map_dir, "depth.hdf5"))
        self.SemanticDB = SemanticFeaturesDB(os.path.join(semantic_map_dir, "siglip2_p32_256.hdf5"))
        self.ObjectTextDB = SingleTextDescriptionDB(
            '/home/files/A/zhanghuaxiang3/GridMM/datasets//R2R/features/object_embeddings-base.pt',
            mode='object'
        )
        self.SpatialTextDB = SingleTextDescriptionDB(
            '/home/files/A/zhanghuaxiang3/GridMM/datasets//R2R/features/spatial_embeddings-base.pt',
            mode='spatial'
        )
        self.viewpoint_info = json.load(open(os.path.join(semantic_map_dir, "viewpoint_info.json")))
        self.cur_vp = None

        self.gt_path = None


    def __len__(self):
        return len(self.data)

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.is_train:
            ran = np.array([random.random() for i in range(36)])
            with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)

            with h5py.File(self.img_ft_file + '/aug_views.hdf5', 'r') as f:
                aug_fts = f[key][...][:].astype(np.float32)
                aug_fts = aug_fts[:, :768]  # 截取前 768 维

            view_fts[ran > 0.5] = aug_fts[ran > 0.5]
        else:
            with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)

        obj_attrs = {}
        obj_fts = np.zeros((0, self.obj_feat_size + self.obj_prob_size), dtype=np.float32)
        if self.obj_ft_file is not None:
            with h5py.File(self.obj_ft_file, 'r') as f:
                if key in f:
                    obj_fts = f[key][...].astype(np.float32)
                    obj_fts = obj_fts[:self.max_objects]
                    for attr_key, attr_value in f[key].attrs.items():
                        if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                            obj_attrs[attr_key] = attr_value[:self.max_objects]
        if self.in_memory:
            self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100  # ignore
            # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = 100000000.
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k  # [stop] is 0
            # local: 
            cand_min_dist = 100000000.
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                                + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1  # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False,
            return_obj_label=False, end_vp=None
    ):
        """
        生成模型输入数据
        
        参数:
            idx (int): 数据索引
            end_vp_type (str): 终点视点类型，可选值为:
                - 'pos': 从正样本视点中选择
                - 'neg_in_gt_path': 从真实路径中的非正样本视点选择
                - 'neg_others': 从其他非路径视点中选择
            return_img_probs (bool): 是否返回图像概率特征
            return_act_label (bool): 是否返回动作标签
            return_obj_label (bool): 是否返回物体标签
            end_vp (str): 指定的终点视点ID，如果为None则根据end_vp_type选择
            
        返回:
            dict: 包含模型输入所需的所有特征和标签
        """
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        self.cur_vp = start_vp
        self.heading = start_heading
        pos_vps = item['pos_vps']  # 正样本视点列表
        gt_path = item['path']  # 真实路径

        # 根据end_vp_type选择终点视点
        if end_vp is None:
            if end_vp_type == 'pos':
                # 从正样本视点中随机选择
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                # 从真实路径中的非正样本视点随机选择
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                # 从其他非路径视点随机选择
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        # 计算从起点到终点的最短路径
        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        self.gt_path = gt_path
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        # 如果路径过长，进行截断
        if len(gt_path) > TRAIN_MAX_STEP:
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]

        # 获取轨迹全景特征
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids, grid_spatial_text,grid_fts, grid_map, gridmap_pos_fts, target_patch_id, traj_text_feats = self.get_traj_pano_fts(
            scan, gt_path)

        # 获取全局地图输入特征
        # global: 第一个token是[stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # 获取局部视点位置特征
        # local: 第一个token是[stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
                                         traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        # 构建输出字典
        outs = {
            'instr_id': item['instr_id'],  # 指令ID
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],  # 指令编码

            # 轨迹特征
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],  # 视图图像特征
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],  # 物体图像特征
            'traj_loc_fts': traj_loc_fts,  # 位置特征
            'traj_nav_types': traj_nav_types,  # 导航类型
            'traj_cand_vpids': traj_cand_vpids,  # 候选视点ID
            'traj_vpids': gt_path,  # 路径视点ID序列

            # 图像描述特征
            'traj_text_feats': traj_text_feats,  # 视图图像描述特征

            # 全局地图特征
            'gmap_vpids': gmap_vpids,  # 全局地图视点ID
            'gmap_step_ids': gmap_step_ids,  # 步数ID
            'gmap_visited_masks': gmap_visited_masks,  # 访问掩码
            'gmap_pos_fts': gmap_pos_fts,  # 位置特征
            'gmap_pair_dists': gmap_pair_dists,  # 视点间距离

            # 视点特征
            'vp_pos_fts': vp_pos_fts,  # 视点位置特征
            'vp_angles': last_vp_angles,  # 视点角度

            # 网格地图特征
            'grid_spatial_text': grid_spatial_text,  # 网格位置和文本特征
            'grid_fts': grid_fts,  # 网格特征
            'grid_map': grid_map,  # 网格地图
            'gridmap_pos_fts': gridmap_pos_fts,  # 网格位置特征
            'target_patch_id': target_patch_id  # 目标网格ID
        }

        # 根据需要添加额外输出
        if return_obj_label:
            # 添加物体标签
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            # 添加动作标签
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label  # 全局动作标签
            outs['local_act_labels'] = local_act_label  # 局部动作标签

        if return_img_probs:
            # 添加图像概率
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)  # 视图概率
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)  # 物体概率

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        """
        计算当前位置的视角方向（水平角度和垂直角度）
        
        参数:
            scan (str): 场景ID
            path (list): 导航路径，包含视点ID序列
            start_heading (float): 起始水平角度（弧度）
            
        返回:
            tuple: (heading, elevation)
                - heading (float): 水平角度（弧度），范围[0, 2π]
                - elevation (float): 垂直角度（弧度），范围[-π/2, π/2]
                
        说明:
            - 如果路径长度小于2，使用起始角度和0垂直角度
            - 否则，根据前一个视点到当前视点的方向计算角度
            - 水平角度每30度一个离散值（共12个）
            - 垂直角度每30度一个离散值（上中下3个）
        """
        if len(path) < 2:
            # 路径太短，使用起始角度
            heading = start_heading
            elevation = 0
        else:
            # 根据前一个视点到当前视点的方向计算角度
            prev_vp = path[-2]  # 前一个视点
            cur_vp = path[-1]  # 当前视点
            viewidx = self.scanvp_cands['%s_%s' % (scan, prev_vp)][cur_vp][0]  # 视角索引
            heading = (viewidx % 12) * math.radians(30)  # 水平角度，每30度一个
            elevation = (viewidx // 12 - 1) * math.radians(30)  # 垂直角度，-30/0/30度
        return heading, elevation


    def get_gridmap_pos_fts(self, half_len):
        rel_angles, rel_dists = [], []
        center_position = [0., 0., 0.]

        cell_len = half_len * 2 / GLOBAL_WIDTH
        for i in range(GLOBAL_WIDTH):
            for j in range(GLOBAL_HEIGHT):
                position = [0., 0., 0.]
                position[0] = i * cell_len - half_len + cell_len / 2.
                position[1] = j * cell_len - half_len + cell_len / 2.
                position[2] = 0.
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(center_position, position,
                                                                                base_heading=0., base_elevation=0.)
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST]
                )


        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1])
        gridmap_pos_fts = np.concatenate([rel_ang_fts, rel_dists], 1)

        return gridmap_pos_fts

    def getGlobalMap(self, scan_id, viewpoint_id):


        viewpoint_x_list = []
        viewpoint_y_list = []
        depth = self.DepthDB.get_image_feature(scan_id, viewpoint_id)
        patch_center_index = np.array([8 + i * 16 for i in range(8)])

        depth = depth[:, patch_center_index][:, :, patch_center_index].reshape(36, -1)

        depth_mask = np.ones(depth.shape)
        depth_mask[depth == 0] = 0
        self.global_mask.append(depth_mask[12:24].reshape(12, -1))
        position = self.viewpoint_info['%s_%s' % (scan_id, viewpoint_id)]

        cur_step_id = len(self.global_mask) - 1
        next_id = cur_step_id
        if cur_step_id < len(self.gt_path) - 1:
            next_id = cur_step_id + 1

        target_position_x = self.viewpoint_info['%s_%s' % (scan_id, self.gt_path[next_id])]["x"] - position["x"]
        target_position_y = self.viewpoint_info['%s_%s' % (scan_id, self.gt_path[next_id])]["y"] - position["y"]

        for ix in range(12, 24):
            rel_x, rel_y = get_rel_position(depth[ix:ix + 1], (ix - 12) * math.pi / 6)
            global_x = rel_x + position["x"]
            global_y = rel_y + position["y"]
            viewpoint_x_list.append(global_x)
            viewpoint_y_list.append(global_y)

        semantic = self.SemanticDB.get_image_feature(scan_id, viewpoint_id)
        SpatialText=self.SpatialTextDB.get_text_features(scan_id, viewpoint_id)
        SpatialText = SpatialText[12:24]  # 只保留中间水平12个
        if self.global_semantic is None or not isinstance(self.global_semantic,
                                                          np.ndarray) or self.global_semantic.size == 0:
            # print(f"semantic.shape: {semantic.shape}")
            self.global_semantic = semantic
            self.global_spatial_text = SpatialText
            self.global_map = np.zeros((12 * 64,))


        else:
            # print(f"semantic.shape: {semantic.shape}")
            self.global_semantic = semantic
            self.global_spatial_text = SpatialText
            self.global_map = np.concatenate([self.global_map, np.zeros((12 * 64,))], 0)

        self.global_map.fill(-1)
        position_x = np.concatenate(viewpoint_x_list, 0)
        position_y = np.concatenate(viewpoint_y_list, 0)
        self.global_position_x.append(position_x)
        self.global_position_y.append(position_y)

        tmp_max_x = position_x.max()
        if tmp_max_x > self.max_x: self.max_x = tmp_max_x
        tmp_min_x = position_x.min()
        if tmp_min_x < self.min_x: self.min_x = tmp_min_x
        tmp_max_y = position_y.max()
        if tmp_max_y > self.max_y: self.max_y = tmp_max_y
        tmp_min_y = position_y.min()
        if tmp_min_y < self.min_y: self.min_y = tmp_min_y

        if position["x"] - self.min_x > self.max_x - position["x"]:
            x_half_len = position["x"] - self.min_x
        else:
            x_half_len = self.max_x - position["x"]

        if position["y"] - self.min_y > self.max_y - position["y"]:
            y_half_len = position["y"] - self.min_y
        else:
            y_half_len = self.max_y - position["y"]

        if x_half_len > y_half_len:
            half_len = x_half_len
        else:
            half_len = y_half_len

        half_len = half_len * 2 /3
        min_x = position["x"] - half_len
        max_x = position["x"] + half_len
        min_y = position["y"] - half_len
        max_y = position["y"] + half_len

        angle = -self.heading
        sRotatex = target_position_x * math.cos(angle) + target_position_y * math.sin(angle)
        sRotatey = target_position_y * math.cos(angle) - target_position_x * math.sin(angle)

        target_patch_x = int((sRotatex + half_len) * 16 // (2 * half_len))
        target_patch_y = int((sRotatey + half_len) * 16 // (2 * half_len))
        target_patch_x = min(max(target_patch_x, 0), 15)
        target_patch_y = min(max(target_patch_y, 0), 15)
        target_patch_id = 1 + target_patch_x * 16 + target_patch_y

        if next_id == cur_step_id:
            target_patch_id = 0

        global_position_x = np.concatenate(self.global_position_x, 0)
        global_position_y = np.concatenate(self.global_position_y, 0)
        local_map = self.global_semantic
        global_mask = np.concatenate(self.global_mask, 0)

        tmp_x = global_position_x - position["x"]
        tmp_y = global_position_y - position["y"]

        map_x = tmp_x * math.cos(angle) + tmp_y * math.sin(angle)
        map_y = tmp_y * math.cos(angle) - tmp_x * math.sin(angle)
        map_x = ((map_x + half_len) / (2 * half_len) * (GLOBAL_WIDTH - 1)).astype(np.int32)

        map_y = ((map_y + half_len) / (2 * half_len) * (GLOBAL_HEIGHT - 1)).astype(np.int32)

        map_x[map_x < 0] = 0
        map_x[map_x >= GLOBAL_WIDTH] = GLOBAL_WIDTH -1

        map_y[map_y < 0] = 0
        map_y[map_y >= GLOBAL_HEIGHT] = GLOBAL_HEIGHT -1

        label_index = (global_mask == 1)

        map_index = map_x * 16 + map_y
        map_index = map_index.reshape(-1)
        label_index = label_index.reshape(-1)

        for patch_id in range(GLOBAL_WIDTH * GLOBAL_HEIGHT):
            filter_index = (map_index == patch_id) & label_index

            self.global_map[filter_index] = patch_id

        gridmap_pos_fts = self.get_gridmap_pos_fts(half_len)
        return self.global_spatial_text,self.global_semantic, self.global_position_x, self.global_position_y, self.global_mask, self.global_map, self.max_x, self.min_x, self.max_y, self.min_y, gridmap_pos_fts, target_patch_id

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''

        self.global_semantic = []
        self.global_spatial_text=[]
        self.global_position_x = []
        self.global_position_y = []
        self.global_mask = []
        self.max_x = -10000
        self.min_x = 10000
        self.max_y = -10000
        self.min_y = 10000
        self.global_map = None

        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []
        traj_text_feats = []  # === New ===
        grid_map = None
        grid_fts = np.array([])
        grid_spatial_text = []
        for vp in path:
            if self.cur_vp != vp and vp in self.scanvp_cands['%s_%s' % (scan, self.cur_vp)]:
                viewidx = self.scanvp_cands['%s_%s' % (scan, self.cur_vp)][vp][0]
                self.heading = (viewidx % 12) * math.radians(30)
            self.cur_vp = vp
            self.global_spatial_text,self.global_semantic, self.global_position_x, self.global_position_y, self.global_mask, self.global_map, self.max_x, self.min_x, self.max_y, self.min_y, gridmap_pos_fts, target_patch_id = self.getGlobalMap(
                scan, vp)
            if grid_fts.shape == (0,):
                grid_fts = self.global_semantic
                global_spatial_array = np.stack([t.cpu().numpy() for t in self.global_spatial_text], axis=0)
                grid_spatial_text = [global_spatial_array]
            else:
                grid_fts = np.concatenate((grid_fts, self.global_semantic), axis=0)
                global_spatial_array = np.stack([t.cpu().numpy() for t in self.global_spatial_text], axis=0)
                grid_spatial_text.append(global_spatial_array)

            grid_map = self.global_map

            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)
            object_embeds = self.ObjectTextDB.get_text_features(scan, vp)

            view_img_fts, view_angles, cand_vpids, view_text_feats= [], [], [],[]  # === track indices ===
            # cand views
            nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_text_feats.append(object_embeds[v[0]])

                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non-can
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            view_text_feats.extend([object_embeds[idx].cpu().numpy() for idx in range(36) if idx not in used_viewidxs])
            view_text_feats = np.stack(view_text_feats, 0)
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h / self.obj_image_h, w / self.obj_image_w, (h * w) / self.obj_image_size]
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_text_feats.append(view_text_feats)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)
        grid_spatial_text=np.stack(grid_spatial_text,axis=0)
        #print(f"[DEBUG] grid_spatial_text.shape = {grid_spatial_text.shape}")
        #print(f"[DEBUG] gridfts.shape = {grid_fts.shape}")
        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids, grid_spatial_text,grid_fts.reshape(
            (-1, 768)), grid_map, gridmap_pos_fts, target_patch_id, traj_text_feats
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        """
        获取全局地图的输入特征
        
        参数:
            scan (str): 场景ID
            path (list): 视点ID序列，表示导航路径
            cur_heading (float): 当前水平角度（弧度）
            cur_elevation (float): 当前垂直角度（弧度）
            
        返回:
            tuple: 包含以下全局地图特征：
            - gmap_vpids (list): 全局地图中的视点ID列表
            - gmap_step_ids (list): 步数ID列表，表示每个视点在路径中的位置
            - gmap_visited_masks (list): 访问掩码列表，表示每个视点是否已访问
            - gmap_pos_fts (np.ndarray): 位置特征数组，包含相对位置和方向信息
            - gmap_pair_dists (np.ndarray): 视点对之间的距离矩阵
            
        说明:
            该方法构建全局地图表示，包括：
            1. 已访问的视点
            2. 当前视点的邻居视点
            3. 视点之间的相对位置和距离
            4. 视点的访问状态
            
            全局地图用于帮助模型理解整体环境结构和导航历史。
        """
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s' % (scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)

        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i + 1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists

    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                     (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)

        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len + 1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts

        return vp_pos_fts


class R2RTextPathData(ReverieTextPathData):
    def __init__(
            self, anno_files, img_ft_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=768, image_prob_size=1000, angle_feat_size=4,
            max_txt_len=100, in_memory=True, is_train=False, act_visited_node=False
    ):
        super().__init__(
            anno_files, img_ft_file, None, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0, 
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory, is_train=is_train,
            act_visited_node=act_visited_node
        )

    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.is_train:
            ran = np.array([random.random() for i in range(36)])
            with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)
            with h5py.File(self.img_ft_file + '/aug_views.hdf5', 'r') as f:
                aug_fts = f[key][...][:].astype(np.float32)
                aug_fts = aug_fts[:, :768]  # 截取前 768 维

            view_fts[ran > 0.5] = aug_fts[ran > 0.5]
        else:
            with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)

        return view_fts

    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1  # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        self.cur_vp = start_vp
        start_heading = item['heading']
        self.heading = start_heading
        gt_path = item['path']
        self.gt_path = gt_path
        if end_vp is None:
            if end_vp_type == 'pos': 
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)

        gt_path = gt_path[:end_idx + 1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]

        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, grid_spatial_text,grid_fts, grid_map, gridmap_pos_fts, target_patch_id, traj_text_feats = self.get_traj_pano_fts(
            scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
                                         traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        # 构建模型输入和标签的字典
        outs = {
            # 指令相关字段
            'instr_id': item['instr_id'],  # 当前导航指令的唯一标识符
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],  # 编码后的指令文本(截断到最大长度)

            # 轨迹视图特征
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],  # 轨迹中每个视角的图像特征
            'traj_loc_fts': traj_loc_fts,  # 轨迹位置特征(角度+框特征)
            'traj_nav_types': traj_nav_types,  # 导航类型标记(1=候选视角,0=非候选视角)
            'traj_cand_vpids': traj_cand_vpids,  # 每个位置的候选视点ID列表
            'traj_vpids': gt_path,  # 真实的轨迹视点ID序列

            # 图像描述特征
            'traj_text_feats': traj_text_feats,  # 视图图像描述特征

            # 全局地图相关特征
            'gmap_vpids': gmap_vpids,  # 全局地图中的视点ID列表
            'gmap_step_ids': gmap_step_ids,  # 每个视点在全局地图中的步数ID
            'gmap_visited_masks': gmap_visited_masks,  # 标记哪些视点已经被访问过
            'gmap_pos_fts': gmap_pos_fts,  # 全局地图中视点的位置特征
            'gmap_pair_dists': gmap_pair_dists,  # 全局地图中视点间的距离矩阵

            # 当前视点相关特征
            'vp_pos_fts': vp_pos_fts,  # 当前视点的位置特征
            'vp_angles': last_vp_angles,  # 当前视点的角度信息

            # 网格地图相关特征 (用于基于网格的导航方法)
            'grid_spatial_text': grid_spatial_text,  # 网格位置和文本特征
            'grid_fts': grid_fts,  # 网格特征
            'grid_map': grid_map,  # 网格地图表示
            'gridmap_pos_fts': gridmap_pos_fts,  # 网格位置特征
            'target_patch_id': target_patch_id  # 目标位置的网格ID
        }

        # 如果需要返回动作标签(训练时使用)
        if return_act_label:
            # 获取全局和局部动作标签
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label  # 全局动作标签(在全局地图中选择下一个视点)
            outs['local_act_labels'] = local_act_label  # 局部动作标签(在当前视点的候选中选择)

        # 如果需要返回图像概率(某些模型可能需要视觉分类概率)
        if return_img_probs:
            # 注意: DINOv2特征可能不包含分类概率，这部分可能需要修改或移除
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)

        return outs  # 返回完整的输入字典

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''

        self.global_semantic = []
        self.global_spatial_text=[]
        self.global_position_x = []
        self.global_position_y = []
        self.global_mask = []
        self.max_x = -10000
        self.min_x = 10000
        self.max_y = -10000
        self.min_y = 10000


        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], []
        traj_text_feats = []  # === New ===
        grid_map = None
        grid_fts = np.array([])
        grid_spatial_text = []
        for vp in path:
            if self.cur_vp != vp and vp in self.scanvp_cands['%s_%s' % (scan, self.cur_vp)]:
                viewidx = self.scanvp_cands['%s_%s' % (scan, self.cur_vp)][vp][0]
                self.heading = (viewidx % 12) * math.radians(30)
            self.cur_vp = vp
            self.global_spatial_text,self.global_semantic, self.global_position_x, self.global_position_y, self.global_mask, self.global_map, self.max_x, self.min_x, self.max_y, self.min_y, gridmap_pos_fts, target_patch_id = self.getGlobalMap(
                scan, vp)
            if grid_fts.shape == (0,):
                grid_fts = self.global_semantic
                global_spatial_array = np.stack([t.cpu().numpy() for t in self.global_spatial_text], axis=0)
                grid_spatial_text = [global_spatial_array]
            else:
                grid_fts = np.concatenate((grid_fts, self.global_semantic), axis=0)
                global_spatial_array = np.stack([t.cpu().numpy() for t in self.global_spatial_text], axis=0)
                grid_spatial_text.append(global_spatial_array)

            grid_map = self.global_map

            view_fts = self.get_scanvp_feature(scan, vp)
            object_embeds = self.ObjectTextDB.get_text_features(scan, vp)  # [36] List[Tensor[768]]

            view_img_fts, view_angles, cand_vpids, view_text_feats= [], [], [],[]  # === track indices ===
            # cand views
            nav_cands = self.scanvp_cands['%s_%s' % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_text_feats.append(object_embeds[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
                # non-can
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            view_text_feats.extend([object_embeds[idx].cpu().numpy() for idx in range(36) if idx not in used_viewidxs])
            view_text_feats = np.stack(view_text_feats, 0)
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_text_feats.append(view_text_feats)
            traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))
            traj_cand_vpids.append(cand_vpids)

            last_vp_angles = view_angles
        grid_spatial_text = np.stack(grid_spatial_text, axis=0)  # 形状 (T, 36, 768)
        #print(f"[DEBUG] grid_spatial_text.shape = {grid_spatial_text.shape}")
        #print(f"[DEBUG] gridfts.shape = {grid_fts.shape}")
        return traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles, grid_spatial_text,grid_fts.reshape(
            (-1, 768)), grid_map, gridmap_pos_fts, target_patch_id, traj_text_feats


class SoonTextPathData(ReverieTextPathData):
    def __init__(
            self, anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
            obj_feat_size=None, obj_prob_size=None, max_objects=20,
            max_txt_len=100, in_memory=True, is_train=False, act_visited_node=False
    ):
        super().__init__(
            anno_files, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=obj_feat_size,
            obj_prob_size=obj_prob_size, max_objects=max_objects,
            max_txt_len=max_txt_len, in_memory=in_memory, is_train=is_train,
            act_visited_node=act_visited_node
        )
        self.obj_image_h = self.obj_image_w = 600
        self.obj_image_size = 600 * 600

    def get_scanvp_feature(self, scan, viewpoint):

        key = '%s_%s' % (scan, viewpoint)

        if self.is_train:
            ran = np.array([random.random() for i in range(36)])
            with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)

            with h5py.File(self.img_ft_file + '/aug_views.hdf5', 'r') as f:
                aug_fts = f[key][...][:].astype(np.float32)
                aug_fts = aug_fts[:, :768]  # 截取前 768 维

            view_fts[ran > 0.5] = aug_fts[ran > 0.5]
        else:
            with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                view_fts = f[key][...][:].astype(np.float32)

        obj_attrs = {}
        obj_fts = np.zeros((0, self.obj_feat_size + self.obj_prob_size), dtype=np.float32)
        if self.obj_ft_file is not None:
            with h5py.File(self.obj_ft_file, 'r') as f:
                if key in f:
                    obj_fts = f[key][...].astype(np.float32)
                    obj_fts = obj_fts[:self.max_objects]
                    for attr_key, attr_value in f[key].attrs.items():
                        if attr_key in ['directions', 'bboxes', 'obj_ids']:
                            obj_attrs[attr_key] = attr_value[:self.max_objects]
                    obj_attrs['bboxes'] = np.array(obj_attrs['bboxes']).astype(np.float32)
                    obj_attrs['sizes'] = np.zeros((len(obj_attrs['bboxes']), 2), dtype=np.float32)
                    obj_attrs['sizes'][:, 0] = obj_attrs['bboxes'][:, 2] - obj_attrs['bboxes'][:, 0]
                    obj_attrs['sizes'][:, 1] = obj_attrs['bboxes'][:, 3] - obj_attrs['bboxes'][:, 1]
        if self.in_memory:
            self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        obj_label = item['obj_pseudo_label']['idx']
        if obj_label >= self.max_objects:
            obj_label = -100
        return obj_label

    def get_input(
            self, idx, end_vp_type, return_img_probs=False, return_act_label=False,
            return_obj_label=False, end_vp=None
    ):
        if end_vp_type == 'pos':
            end_vp = self.data[idx]['path'][-1]
        return super().get_input(
            idx, end_vp_type,
            return_img_probs=return_img_probs,
            return_act_label=return_act_label,
            return_obj_label=return_obj_label, 
            end_vp=end_vp
        )
