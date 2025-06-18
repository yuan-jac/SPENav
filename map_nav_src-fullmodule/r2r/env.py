''' Batched REVERIE navigation environment '''

import json
import math
import os
import random
from collections import defaultdict

import MatterSim
import h5py
import networkx as nx
import numpy as np
import torch

from map_nav_src.r2r.eval_utils import cal_dtw, cal_cls
from map_nav_src.utils.data import angle_feature, get_all_point_angle_feature
from map_nav_src.utils.data import load_nav_graphs, new_simulator


def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint

WIDTH = 128
HEIGHT = 128
VFOV = 60
GLOBAL_WIDTH = 16
GLOBAL_HEIGHT = 16

ERROR_MARGIN = 3.0
MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20


def get_angle_fts(headings, elevations, angle_feat_size=4):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist


class DepthFeaturesDB(object):
    def __init__(self, img_ft_file):
        self.image_feat_size = WIDTH*HEIGHT
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)

        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.uint16)
                self._feature_store[key] = ft
        return ft


class SemanticFeaturesDB(object):
    def __init__(self, img_ft_file):
        self.image_feat_size = 8*8
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)

        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float16)
                self._feature_store[key] = ft
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



class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, semantic_map_dir="datasets/R2R/features", batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.batch_size = batch_size
        self.sims = []
        self.global_semantic = [[] for i in range(batch_size)]
        self.spatial_text_fts = [[] for i in range(batch_size)]
        self.global_position_x = [[] for i in range(batch_size)]
        self.global_position_y = [[] for i in range(batch_size)]
        self.global_mask = [[] for i in range(batch_size)]
        self.max_x = [-10000 for i in range(batch_size)]
        self.min_x = [10000 for i in range(batch_size)]
        self.max_y = [-10000 for i in range(batch_size)]
        self.min_y = [10000 for i in range(batch_size)]
        self.heading = [0 for i in range(batch_size)]
        self.global_map = [[] for i in range(batch_size)]

        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir:
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setPreloadingEnabled(True)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setBatchSize(1)
            sim.initialize()
            self.sims.append(sim)

        self.DepthDB = DepthFeaturesDB(os.path.join(semantic_map_dir,"depth.hdf5"))
        self.SemanticDB = SemanticFeaturesDB(os.path.join(semantic_map_dir,"siglip2_p32_256.hdf5"))
        self.ObjectTextDB = SingleTextDescriptionDB(
            'datasets/R2R/features/object_embeddings-base.pt',
            mode='object'
        )
        self.SpatialTextDB = SingleTextDescriptionDB(
            'datasets/R2R/features/spatial_embeddings-base.pt',
            mode='spatial'
        )
        self.viewpoint_info = json.load(open(os.path.join(semantic_map_dir,"viewpoint_info.json")))
        self.feature_states = [None for i in range(len(self.sims))]




    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):

        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])
            self.heading[i] = heading

        self.global_semantic = [[] for i in range(self.batch_size)]
        self.spatial_text_fts = [[] for i in range(self.batch_size)]
        self.global_position_x = [[] for i in range(self.batch_size)]
        self.global_position_y = [[] for i in range(self.batch_size)]
        self.global_mask = [[] for i in range(self.batch_size)]
        self.max_x = [-10000 for i in range(self.batch_size)]
        self.min_x = [10000 for i in range(self.batch_size)]
        self.max_y = [-10000 for i in range(self.batch_size)]
        self.min_y = [10000 for i in range(self.batch_size)]
        self.heading = [0 for i in range(self.batch_size)]
        self.global_map = [[] for i in range(self.batch_size)]
        self.feature_states = [None for i in range(len(self.sims))]


    def get_global_target(self,obs,next_gt_vp):

        target_patch_ids = []
        for i in range(len(next_gt_vp)):

            scan = obs[i]['scan']
            cur_vp = obs[i]['viewpoint']
            next_vp = next_gt_vp[i]

            if next_vp == None:
                target_patch_ids.append(0)
                continue

            position = self.viewpoint_info['%s_%s' % (scan, cur_vp)]

            target_position_x = self.viewpoint_info['%s_%s' % (scan, next_vp)]["x"] - position["x"]
            target_position_y = self.viewpoint_info['%s_%s' % (scan, next_vp)]["y"] - position["y"]

            if position["x"]-self.min_x[i] > self.max_x[i]-position["x"] : x_half_len = position["x"]-self.min_x[i]
            else: x_half_len = self.max_x[i]-position["x"]

            if position["y"]-self.min_y[i] > self.max_y[i]-position["y"] : y_half_len = position["y"]-self.min_y[i]
            else: y_half_len = self.max_y[i]-position["y"]

            if x_half_len > y_half_len : half_len = x_half_len
            else: half_len = y_half_len

            half_len = half_len * 2/3
            min_x = position["x"] - half_len
            max_x = position["x"] + half_len
            min_y = position["y"] - half_len
            max_y = position["y"] + half_len

            angle = -self.heading[i]
            sRotatex = target_position_x * math.cos(angle) + target_position_y * math.sin(angle)
            sRotatey = target_position_y * math.cos(angle) - target_position_x * math.sin(angle)

            target_patch_x = int((sRotatex + half_len)*16 // (2*half_len))
            target_patch_y = int((sRotatey + half_len)*16 // (2*half_len))
            target_patch_x = min(max(target_patch_x,0),15)
            target_patch_y = min(max(target_patch_y,0),15)
            target_patch_id = 1 + target_patch_x*16 + target_patch_y
            target_patch_ids.append(target_patch_id)

        return target_patch_ids

    def get_gridmap_pos_fts(self, half_len):
        rel_angles, rel_dists = [], []
        center_position = [0.,0.,0.]

        cell_len = half_len*2 / GLOBAL_WIDTH
        for i in range(GLOBAL_WIDTH):
            for j in range(GLOBAL_HEIGHT):
                position = [0.,0.,0.]
                position[0] = i*cell_len - half_len + cell_len/2.
                position[1] = j*cell_len - half_len + cell_len/2.
                position[2] = 0.
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(center_position, position, base_heading=0., base_elevation=0.)
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST]
                )


        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1])
        gridmap_pos_fts = np.concatenate([rel_ang_fts, rel_dists], 1)

        return gridmap_pos_fts

    def getGlobalMap(self,tid):
        i = tid
        sim = self.sims[i]
        state = sim.getState()[0]
        scan_id = state.scanId
        viewpoint_id = state.location.viewpointId


        self.heading[i] = state.heading
        viewpoint_x_list = []
        viewpoint_y_list = []
        depth = self.DepthDB.get_image_feature(scan_id,viewpoint_id)
        patch_center_index = np.array([8 + i * 16 for i in range(8)])

        depth = depth[:,patch_center_index][:,:,patch_center_index].reshape(36,-1)

        depth_mask = np.ones(depth.shape)
        depth_mask[depth==0] = 0
        self.global_mask[i].append(depth_mask[12:24].reshape(12,-1))
        position = self.viewpoint_info['%s_%s' % (scan_id, viewpoint_id)]


        for ix in range(12,24):
            rel_x, rel_y = get_rel_position(depth[ix:ix+1],(ix-12)*math.pi/6)
            global_x = rel_x + position["x"]
            global_y = rel_y + position["y"]
            viewpoint_x_list.append(global_x)
            viewpoint_y_list.append(global_y)

        semantic = self.SemanticDB.get_image_feature(scan_id,viewpoint_id)
        SpatialText = self.SpatialTextDB.get_text_features(scan_id, viewpoint_id)
        SpatialText = SpatialText[12:24]  # 只保留中间水平12个

        if len(self.global_semantic[i]) == 0:
            self.global_semantic[i] = semantic.reshape(-1, 768)
            self.spatial_text_fts[i]= SpatialText
            self.global_map[i] = np.zeros((12*64,))
        else:
            self.spatial_text_fts[i]= np.concatenate((self.spatial_text_fts, SpatialText),axis=0)
            self.global_semantic[i] = np.concatenate((self.global_semantic[i], semantic.reshape(-1, 768)), axis=0)
            self.global_map[i] = np.concatenate([self.global_map[i],np.zeros((12*64,))],0)

        self.global_map[i].fill(-1)
        position_x = np.concatenate(viewpoint_x_list,0)
        position_y = np.concatenate(viewpoint_y_list,0)
        self.global_position_x[i].append(position_x)
        self.global_position_y[i].append(position_y)

        tmp_max_x = position_x.max()
        if tmp_max_x > self.max_x[i]: self.max_x[i] = tmp_max_x
        tmp_min_x = position_x.min()
        if tmp_min_x < self.min_x[i]: self.min_x[i] = tmp_min_x
        tmp_max_y = position_y.max()
        if tmp_max_y > self.max_y[i]: self.max_y[i] = tmp_max_y
        tmp_min_y = position_y.min()
        if tmp_min_y < self.min_y[i]: self.min_y[i] = tmp_min_y


        if position["x"]-self.min_x[i] > self.max_x[i]-position["x"] : x_half_len = position["x"]-self.min_x[i]
        else: x_half_len = self.max_x[i]-position["x"]

        if position["y"]-self.min_y[i] > self.max_y[i]-position["y"] : y_half_len = position["y"]-self.min_y[i]
        else: y_half_len = self.max_y[i]-position["y"]

        if x_half_len > y_half_len : half_len = x_half_len
        else: half_len = y_half_len

        half_len = half_len * 2/3
        min_x = position["x"] - half_len
        max_x = position["x"] + half_len
        min_y = position["y"] - half_len
        max_y = position["y"] + half_len

        angle = -self.heading[i]

        global_position_x = np.concatenate(self.global_position_x[i],0)
        global_position_y = np.concatenate(self.global_position_y[i],0)
        local_map = self.global_semantic[i]
        global_mask = np.concatenate(self.global_mask[i],0)

        tmp_x = global_position_x - position["x"]
        tmp_y = global_position_y - position["y"]

        map_x = tmp_x * math.cos(angle) + tmp_y * math.sin(angle)
        map_y = tmp_y * math.cos(angle) - tmp_x * math.sin(angle)
        map_x = ((map_x + half_len) / (2*half_len) * (GLOBAL_WIDTH-1)).astype(np.int32)

        map_y = ((map_y + half_len) / (2*half_len) * (GLOBAL_HEIGHT-1)).astype(np.int32)

        map_x[map_x<0] = 0
        map_x[map_x>=GLOBAL_WIDTH] = GLOBAL_WIDTH-1

        map_y[map_y<0] = 0
        map_y[map_y>=GLOBAL_HEIGHT] = GLOBAL_HEIGHT-1

        label_index = (global_mask==1)

        map_index = map_x*16 + map_y
        map_index = map_index.reshape(-1)
        label_index = label_index.reshape(-1)


        for patch_id in range(GLOBAL_WIDTH*GLOBAL_HEIGHT):

            filter_index = (map_index==patch_id)&label_index
            self.global_map[i][filter_index] = patch_id


        gridmap_pos_fts = self.get_gridmap_pos_fts(half_len)

        return tid, self.spatial_text_fts[i],self.global_semantic[i],self.global_position_x[i],self.global_position_y[i],self.global_mask[i],self.global_map[i],self.max_x[i],self.min_x[i],self.max_y[i],self.min_y[i], gridmap_pos_fts


    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        for i in range(len(self.sims)):
            state = self.sims[i].getState()[0]
            scan_id = state.scanId
            viewpoint_id = state.location.viewpointId
            feature = self.feat_db.get_image_feature(scan_id, viewpoint_id)
            Objecttext_feat = self.ObjectTextDB.get_text_features(scan_id, viewpoint_id)
            self.feature_states[i] = (feature,Objecttext_feat, state)


        for i in range(len(self.sims)):
            state = self.sims[i].getState()[0]
            scan_id = state.scanId
            viewpoint_id = state.location.viewpointId

            tid,self.spatial_text_fts[i], self.global_semantic[i],self.global_position_x[i],self.global_position_y[i],self.global_mask[i],self.global_map[i],self.max_x[i],self.min_x[i],self.max_y[i],self.min_y[i], gridmap_pos_fts = self.getGlobalMap(i)
            self.feature_states[i] += (self.spatial_text_fts[i],self.global_semantic[i],self.global_map[i],gridmap_pos_fts)

        return self.feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class R2RNavBatch(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, view_db, instr_data, connectivity_dir,
        batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None
    ):
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size)
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.name = name

        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))


    def _get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
                for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def size(self):
        return len(self.data)



    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def make_candidate(self, feature, objecttext_feat,scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2).astype(np.float32)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = feature[ix]
                object_feat = objecttext_feat[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)

                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': np.float32(loc_heading),
                            'elevation': np.float32(loc_elevation),
                            "normalized_heading": np.float32(state.heading + loc.rel_heading),
                            "normalized_elevation": np.float32(state.elevation + loc.rel_elevation),
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': np.int32(ix),
                            'distance': np.float32(distance),
                            'idx': np.int32(j + 1),
                            'feature': np.concatenate((visual_feat, angle_feat), -1).astype(np.float32),
                            'objecttext_feat':  np.array(object_feat, dtype=np.float32),
                            'position': (np.float32(loc.x), np.float32(loc.y), np.float32(loc.z)),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = np.int32(c_new['pointId'])
                visual_feat = feature[ix].astype(np.float32)
                #print('type(visual_feat):', type(visual_feat))
                object_feat = objecttext_feat[ix].float()
                #print('object_feat',type(object_feat))
                c_new['heading'] = np.float32(c_new['normalized_heading'] - base_heading)
                c_new['elevation'] = np.float32(c_new['normalized_elevation'] - base_elevation)
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'],self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1).astype(np.float32)
                c_new['objecttext_feat'] = np.array(object_feat, dtype=np.float32)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, ObjectText_feat,state,spatial_fts,grid_fts,grid_map,gridmap_pos_fts) in enumerate(self.env.getStates()):

            item = self.batch[i]
            base_view_id = state.viewIndex
            '''for i, feat in enumerate(feature):
                print(f'[{i}] shape: ', np.array(feat).shape)
            for i, feat in enumerate(ObjectText_feat):
                print(f'[{i}] shape: ', np.array(feat).shape)'''
            # Full features
            candidate = self.make_candidate(feature,ObjectText_feat,state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (np.float32(state.location.x), np.float32(state.location.y), np.float32(state.location.z)),
                'heading' : np.float32(state.heading),
                'elevation' : np.float32(state.elevation),
                'objecttext_feat': ObjectText_feat,
                'feature' : feature.astype(np.float32),
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': [np.int32(i) for i in item['instr_encoding']],
                'gt_path' : item['path'],
                'path_id' : item['path_id'],
                'spatial_fts':torch.tensor(spatial_fts),
                'grid_fts': torch.tensor(grid_fts),
                'grid_map': torch.tensor(grid_map),
                'gridmap_pos_fts': torch.tensor(gridmap_pos_fts)
            }
            # RL reward. The negative distance between the state and the final state
            # There are multiple gt end viewpoints on REVERIE.
            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = np.float32(self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]])
            else:
                ob['distance'] = np.float32(0)

            obs.append(ob)

        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()


    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)
        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics

