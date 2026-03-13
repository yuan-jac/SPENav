import torch

# 加载权重文件
weights = torch.load("*/datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new-ckp-2-nogate/ckpts/model_step_1500.pt", map_location='cpu')

# 判断是完整模型还是 state_dict
if isinstance(weights, dict) and 'state_dict' in weights:
    state_dict = weights['state_dict']
else:
    state_dict = weights  # 通常是 dict：{层名: 参数张量}
for name, param in state_dict.items():
    print(f"{name:60} -> shape: {tuple(param.shape)}")
