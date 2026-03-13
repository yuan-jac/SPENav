import collections
import torch
import torch.nn as nn

from .vlnbert_init import get_vlnbert_models

def print_list_tensor_shapes(batch):
    for key, val in batch.items():
        if isinstance(val, list):
            print(f"\n{key}: list (len = {len(val)})")
            for i, item in enumerate(val):
                if isinstance(item, torch.Tensor):
                    print(f"  [{i}]: {tuple(item.shape)}")
                elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                    print(f"  [{i}]: list/tuple of tensors, first shape: {item[0].shape}")
                else:
                    print(f"  [{i}]: type = {type(item)}")
        else:
            print(f"{key}: {type(val)} -> shape = {getattr(val, 'shape', 'N/A')}")

def print_nested_list_shapes(nested_list, name=""):
    if nested_list is None:
        print(f"{name}: None")
        return
    print(f"{name}: list (len = {len(nested_list)})")
    for i, sublist in enumerate(nested_list):
        if isinstance(sublist, list):
            print(f"  [{i}]: list length = {len(sublist)}")
            if len(sublist) > 0 and isinstance(sublist[0], torch.Tensor):
                print(f"    [0] tensor shape: {sublist[0].shape}")
        else:
            print(f"  [{i}]: type = {type(sublist)}")

class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        self.drop_prob = args.feat_dropout

    def shared_dropout(self, feat_img, feat_text):
        if not self.training or self.drop_prob == 0.:
            return feat_img, feat_text

        # 生成同一掩码
        mask = feat_img.new_empty(feat_img.shape).bernoulli_(1 - self.drop_prob)
        mask = mask / (1 - self.drop_prob)

        feat_img = feat_img * mask
        feat_text = feat_text * mask

        return feat_img, feat_text

    def forward(self, mode, batch):
        """print(f"\n====== Mode: {mode} ======")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {tuple(v.shape)}")
            elif v is None:
                print(f"{k}: None")
            else:
                print(f"{k}: {type(v)}")
        print("====== End Batch Info ======\n")
        #print_list_tensor_shapes(batch)"""
        batch = collections.defaultdict(lambda: None, batch)
        #print_nested_list_shapes(batch['gmap_vpids'], "gmap_vpids")

        if mode == 'language':
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            batch['view_img_fts'], batch['view_text_fts'] = self.shared_dropout(
                batch['view_img_fts'], batch['view_text_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s'%mode)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
