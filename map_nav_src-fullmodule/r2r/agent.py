import math
import sys

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from map_nav_src.models.graph_utils import GraphMap
from map_nav_src.models.model import VLNBert, Critic
from map_nav_src.models.ops import pad_tensors_wgrad
from map_nav_src.utils.ops import pad_tensors, gen_seq_masks
from .agent_base import Seq2SeqAgent


class GMapNavAgent(Seq2SeqAgent):
    """
    GMapNavAgent 类，用于基于图的导航任务。
    继承自 Seq2SeqAgent，实现了导航任务中的模型构建、特征提取、导航策略等功能。
    """
    def _build_model(self):
        """
        构建模型，初始化 VLNBert 和 Critic 模型。
        """
        self.vln_bert = VLNBert(self.args).cuda()  # 初始化 VLNBert 模型并移至 GPU
        self.critic = Critic(self.args).cuda()  # 初始化 Critic 模型并移至 GPU
        # 缓存变量
        self.scanvp_cands = {}

    def _language_variable(self, obs):
        """
        处理语言输入，将指令编码为张量。
        :param obs: 观测数据列表，包含每个样本的指令编码。
        :return: 包含文本 ID 和掩码的字典。
        """
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]  # 获取每个指令的长度

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)  # 初始化指令张量
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)  # 初始化掩码
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']  # 填充指令编码
            mask[i, :seq_lengths[i]] = True  # 设置掩码

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()  # 转换为张量并移至 GPU
        mask = torch.from_numpy(mask).cuda()  # 转换为张量并移至 GPU
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask  # 返回文本 ID 和掩码
        }

    def _panorama_feature_variable(self, obs):
        """
        提取全景特征，包括图像特征和位置特征。
        :param obs: 观测数据列表，包含每个样本的全景特征。
        :return: 包含全景特征的字典。
        """
        # 初始化用于存储批次数据的列表
        batch_view_img_fts,batch_view_text_fts, batch_loc_fts, batch_nav_types =[], [], [], []  # 图像特征、图像描述特征，位置特征和导航类型
        batch_view_lens, batch_cand_vpids = [], []  # 视图长度和候选视图 ID 列表

        # 遍历每个观测数据
        for i, ob in enumerate(obs):
            view_text_fts,view_img_fts, view_ang_fts, nav_types, cand_vpids = [],[], [], [], []  # 初始化当前样本的特征列表
            used_viewidxs = set()  # 用于记录已使用的视图索引

            # 提取候选视图特征
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                # 提取图像描述特征
                view_text_fts.append(cc['objecttext_feat'])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])

            # 提取非候选视图特征
            # 遍历全景特征，提取未被标记为候选视图的特征
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_text_fts.extend(
                [x for k, x in enumerate(ob['objecttext_feat']) if k not in used_viewidxs]
            )
            # 补齐导航类型为 0（表示非候选视图）
            nav_types.extend([0] * (36 - len(used_viewidxs)))

            # 合并候选视图和非候选视图的特征
            view_img_fts = np.stack(view_img_fts, 0)  # 将图像特征堆叠成一个 NumPy 数组 (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)  # 将角度特征堆叠成一个 NumPy 数组
            view_text_fts = np.stack(view_text_fts, 0)  # 将图像描述特征堆叠成一个 NumPy 数组
            # 创建一个简单的框特征（固定值 [1, 1, 1]，表示每个视图的边界框）
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            # 将角度特征和框特征拼接成位置特征
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            # 将当前样本的特征添加到批次列表
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))  # 转换为 PyTorch 张量
            batch_view_text_fts.append(torch.from_numpy(view_text_fts))  # 转换为 PyTorch 张量
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))  # 转换为 PyTorch 张量
            batch_nav_types.append(torch.LongTensor(nav_types))  # 转换为 PyTorch 张量
            batch_cand_vpids.append(cand_vpids)  # 添加候选视图 ID 列表
            batch_view_lens.append(len(view_img_fts))  # 记录当前样本的视图数量

        # 对批次数据进行填充，以对齐长度
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()  # 填充图像特征并移至 GPU
        batch_view_text_fts = pad_tensors(batch_view_text_fts).cuda()  # 填充图像描述特征并移至 GPU
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()  # 填充位置特征并移至 GPU
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()  # 填充导航类型并移至 GPU
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()  # 转换为 PyTorch 张量并移至 GPU

        # 返回包含全景特征的字典
        return {
            'view_img_fts': batch_view_img_fts,  # 图像特征
            'view_text_fts': batch_view_text_fts,  # 图像描述特征
            'loc_fts': batch_loc_fts,  # 位置特征
            'nav_types': batch_nav_types,  # 导航类型
            'view_lens': batch_view_lens,  # 视图长度
            'cand_vpids': batch_cand_vpids,  # 候选视图 ID 列表
        }

    def _nav_gmap_variable(self, obs, gmaps):
        """
        构建导航图（Graph Map）相关的变量。
        该函数用于生成导航图的节点特征、位置特征、访问掩码、节点间距离矩阵等信息，
        用于后续的导航策略计算。

        :param obs: 观测数据列表，每个元素是一个字典，包含当前环境的观测信息。
        :param gmaps: 图结构列表，每个元素是一个 GraphMap 对象，表示当前导航任务的图结构。
        :return: 一个字典，包含导航图相关的变量。
        """
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            # 根据参数决定是否编码完整图
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            # 获取每个节点的步数 ID
            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )  # cuda
            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

                batch_gmap_img_embeds.append(gmap_img_embeds)
                batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
                batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
                batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
                batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
                batch_gmap_vpids.append(gmap_vpids)
                batch_gmap_lens.append(len(gmap_vpids))

            # collate
            batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
            batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
            batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
            batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
            batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
            batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
            'grid_spatial_text': [obs[index]['spatial_fts'].cuda() for index in range(len(obs))],
            'grid_fts': [obs[index]['grid_fts'].cuda() for index in range(len(obs))],
            'grid_map': [obs[index]['grid_map'].cuda() for index in range(len(obs))], 'gridmap_pos_fts': torch.cat(
                [obs[index]['gridmap_pos_fts'].unsqueeze(0).cuda() for index in range(len(obs))], dim=0)
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        """
        构建导航任务中视点（Viewpoint）相关的变量。
        该函数用于生成视点的图像嵌入、位置特征、导航掩码等信息，
        用于后续的导航策略计算。

        :param obs: 观测数据列表，每个元素是一个字典，包含当前环境的观测信息。
        :param gmaps: 图结构列表，每个元素是一个 GraphMap 对象，表示当前导航任务的图结构。
        :param pano_embeds: 全景图像的嵌入表示，形状为 (batch_size, num_views, embed_dim)。
        :param cand_vpids: 候选视点的 ID 列表，每个样本对应一个列表。
        :param view_lens: 每个样本的视图数量。
        :param nav_types: 导航类型张量，形状为 (batch_size, num_views)，候选视点为 1，非候选视点为 0。
        :return: 一个字典，包含视点相关的变量。
        """
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens + 1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0  # Stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                   + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:  # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            grid_logits = nav_outs['grid_logits']
            nav_probs = torch.softmax(nav_logits, 1)

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }

            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                )
                # print(t, nav_logits, nav_targets)
                # stop_logits = nav_logits[nav_targets==0]
                # stop_targets = nav_targets[nav_targets==0]

                # if stop_logits.shape[0] != 0:
                #    ml_loss += self.criterion(nav_logits, nav_targets) + self.criterion(stop_logits,stop_targets) * 2.
                # else:
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())

            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets  # teacher forcing

            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs[
                        'gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample':  # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

                    # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj