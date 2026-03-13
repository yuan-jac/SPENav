import json
import math
import os
import random

import h5py
import networkx as nx
import numpy as np


class ImageFeaturesDB(object):
    """
    图像特征数据库，用于加载和缓存图像特征。

    参数:
        img_ft_file: 图像特征文件的目录路径。
        image_feat_size: 图像特征的维度。
        is_train: 是否为训练模式。
    """
    def __init__(self, img_ft_file, image_feat_size, is_train):
        self.image_feat_size = image_feat_size  # 图像特征的维度
        self.img_ft_file = img_ft_file  # 图像特征文件的目录路径
        self._feature_store_1 = {}  # 缓存特征的字典
        self._feature_store_2 = {}
        self._feature_store_3 = {}
        self._feature_store_4 = {}
        self.is_train = is_train  # 是否为训练模式

    def get_image_feature(self, scan, viewpoint):
        """
        获取指定场景和视角的图像特征。

        参数:
            scan: 场景标识符。
            viewpoint: 视角标识符。

        返回:
            view_fts: 图像特征数组。
        """
        key = '%s_%s' % (scan, viewpoint)  # 构造键值
        ran = np.array([random.random() for i in range(36)])  # 随机数数组

        if self.is_train:  # 如果是训练模式
            # 加载主特征
            if key in self._feature_store_4:
                view_fts = self._feature_store_4[key].astype(np.float32)
            else:
                with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                    view_fts = f[key][:, :self.image_feat_size].astype(np.float32)
                    self._feature_store_4[key] = view_fts.astype(np.float16)

            # 加载增强特征
            if key in self._feature_store_3:
                aug_fts = self._feature_store_3[key].astype(np.float32)
            else:
                with h5py.File(self.img_ft_file + '/aug_views.hdf5', 'r') as f:
                    aug_fts = f[key][:, :self.image_feat_size].astype(np.float32)
                    self._feature_store_3[key] = aug_fts.astype(np.float16)
            view_fts[ran > 0.5] = aug_fts[ran > 0.5]  # 随机替换部分特征

        else:  # 如果是测试模式
            # 加载主特征
            if key in self._feature_store_4:
                view_fts = self._feature_store_4[key].astype(np.float32)
            else:
                with h5py.File(self.img_ft_file + '/dino_features_36_vit_b14.hdf5', 'r') as f:
                    view_fts = f[key][:, :self.image_feat_size].astype(np.float32)
                    self._feature_store_4[key] = view_fts.astype(np.float16)
        return view_fts  # 返回图像特征

def load_nav_graphs(connectivity_dir, scans):
    """
    加载每个场景的导航图。

    参数:
        connectivity_dir: 连接性信息目录，包含每个场景的导航图文件。
        scans: 场景列表，每个场景都有一个对应的导航图文件。

    返回:
        graphs: 场景导航图的字典，键是场景标识符，值是对应的导航图。
    """
    # 定义一个辅助函数，用于计算两个图位置之间的欧几里得距离
    def distance(pose1, pose2):
        """
        计算两个图位置之间的欧几里得距离。

        参数:
            pose1: 第一个位置，是一个字典，包含位置信息。
            pose2: 第二个位置，是一个字典，包含位置信息。

        返回:
            距离值，是两个位置之间的欧几里得距离。
        """
        # 提取位置信息，并计算距离
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    # 初始化一个空字典，用于存储每个场景的导航图
    graphs = {}
    # 遍历每个场景
    for scan in scans:
        # 打开当前场景的导航图文件
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            # 创建一个无向图对象
            G = nx.Graph()
            # 初始化一个空字典，用于存储每个节点的位置信息
            positions = {}
            # 加载导航图文件的内容
            data = json.load(f)
            # 遍历导航图中的每个节点
            for i, item in enumerate(data):
                # 如果当前节点被包含在导航图中
                if item['included']:
                    # 遍历当前节点的所有连接
                    for j, conn in enumerate(item['unobstructed']):
                        # 如果当前连接是可达的，并且目标节点也被包含在导航图中
                        if conn and data[j]['included']:
                            # 记录当前节点的位置信息
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]])
                            # 断言目标节点到当前节点的连接也是可达的，确保图是无向的
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            # 在图中添加一条边，连接当前节点和目标节点，并设置边的权重为两个节点之间的距离
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            # 将位置信息设置为图的节点属性
            nx.set_node_attributes(G, values=positions, name='position')
            # 将当前场景的导航图存储到字典中
            graphs[scan] = G
    # 返回包含所有场景导航图的字典
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    """
    获取指定视角的角度特征。

    参数:
        sim: MatterSim 模拟器对象，用于模拟环境中的视角变化。
        angle_feat_size: 角度特征的维度。
        baseViewId: 基础视角 ID（默认为 0），用于计算相对角度。

    返回:
        feature: 一个 (36, angle_feat_size) 的数组，表示每个视角的角度特征。
    """
    # 初始化一个空数组，用于存储每个视角的角度特征
    feature = np.empty((36, angle_feat_size), np.float32)

    # 计算基础视角的方向信息
    base_heading = (baseViewId % 12) * math.radians(30)  # 基础视角的水平方向（heading）
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)  # 基础视角的垂直方向（elevation）

    # 遍历所有 36 个视角
    for ix in range(36):
        # 如果是第一个视角，初始化模拟器的状态
        if ix == 0:
            # 初始化模拟器，设置初始视角
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        # 如果视角索引是 12 的倍数，表示需要切换到下一个垂直层级
        elif ix % 12 == 0:
            # 向上或向下移动视角
            sim.makeAction([0], [1.0], [1.0])
        else:
            # 在当前垂直层级内水平移动视角
            sim.makeAction([0], [1.0], [0])

        # 获取当前视角的状态
        state = sim.getState()[0]
        # 确保当前视角的索引正确
        assert state.viewIndex == ix

        # 计算当前视角的相对方向
        heading = state.heading - base_heading  # 相对水平方向
        elevation = state.elevation - base_elevation  # 相对垂直方向

        # 计算并存储当前视角的角度特征
        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)

    # 返回所有视角的角度特征
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    """
    获取所有视角的角度特征。

    参数:
        sim: MatterSim 模拟器对象，用于模拟环境中的视角变化。
        angle_feat_size: 角度特征的维度。

    返回:
        一个包含所有视角角度特征的列表，每个元素是一个 (36, angle_feat_size) 的数组。
    """
    # 使用列表推导式，调用 get_point_angle_feature 函数计算每个基础视角的角度特征
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

