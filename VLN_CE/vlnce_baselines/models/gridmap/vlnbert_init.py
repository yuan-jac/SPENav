import torch

def get_vlnbert_models(config=None, DATASET='R2R'):
    """
    加载 SPENav 定制 VLN-BERT 模型（R2R / RxR）。
    完全绕过 transformers.from_pretrained 的 NoneType 错误。
    """
    print("🧭 [DEBUG] Using manually loaded VLN-BERT (SPENav version)")

    from transformers import PretrainedConfig
    from .vilmodel import GlocalTextPathNavCMT

    # -------------------------------
    # 选择 checkpoint 路径
    # -------------------------------
    if DATASET == 'R2R':
        model_name_or_path = 'data/pretrained_models/grid_map-models/grid_map.pt'
    elif DATASET == 'RxR':
        model_name_or_path = 'data/pretrained_models/grid_map-models/grid_map_rxr.pt'
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

    # -------------------------------
    # 加载 checkpoint 权重
    # -------------------------------
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        # 针对不同保存格式处理
        if 'vln_bert' in ckpt_weights:
            ckpt_weights = ckpt_weights['vln_bert']['state_dict']
        if 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']

        for k, v in ckpt_weights.items():
            orig_k = k
            if k.startswith('net.'):
                k = k[4:]
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('bert.'):
                k = k[5:]
            if k.startswith('vln_bert.'):
                k = k[9:]
            new_ckpt_weights[k] = v

    # -------------------------------
    # 配置 VLN-BERT
    # -------------------------------
    if DATASET == 'R2R':
        vis_config = PretrainedConfig.from_pretrained('bert-base-uncased')
    elif DATASET == 'RxR':
        vis_config = PretrainedConfig.from_pretrained('*/xlm-roberta-base')

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 768
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 9
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4

    vis_config.fix_lang_embedding = False
    vis_config.fix_pano_embedding = False
    vis_config.fix_local_branch = False
    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1

    # -------------------------------
    # 实例化模型（不使用 .from_pretrained）
    # -------------------------------
    visual_model = GlocalTextPathNavCMT(config=vis_config)

    if new_ckpt_weights:
        missing, unexpected = visual_model.load_state_dict(new_ckpt_weights, strict=False)
        print(f"[VLNBERT] Loaded weights: {len(new_ckpt_weights)} tensors | "
              f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    # -------------------------------
    # 加载 CLIP 权重
    # -------------------------------
    clip_path = 'data/pretrained_models/grid_map-models/ViT-B-32.pt'
    state_dict = torch.jit.load(clip_path, map_location='cpu').state_dict()
    visual_model.clip.load_state_dict(state_dict, strict=False)

    # -------------------------------
    # 加载视觉编码器权重
    # -------------------------------
    vit_path = 'data/pretrained_models/grid_map-models/dino_vitbase16_pretrain.pth'
    state_dict = torch.load(vit_path, map_location='cpu')
    visual_model.visual_encoder.load_state_dict(state_dict)

    print("✅ VLN-BERT (SPENav) 模型加载完成")
    return visual_model
