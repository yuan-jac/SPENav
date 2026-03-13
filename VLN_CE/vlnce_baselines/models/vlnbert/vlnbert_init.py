import os
import torch
from transformers import BertConfig
from vlnce_baselines.models.vlnbert.vlnbert_PREVALENT import VLNBert

def get_vlnbert_models(config=None, DATASET=None):
    model_path = 'data/pretrained_models/rec_vln_bert-models/vlnbert_prevalent_model.bin'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Pretrained VLN-BERT weights not found: {model_path}")

    # Build config manually (VLNBert uses Bert backbone but has visual extensions)
    vis_config = BertConfig.from_pretrained('bert-base-uncased')
    vis_config.img_feature_dim = 2176
    vis_config.img_feature_type = ""
    vis_config.vl_layers = 4
    vis_config.la_layers = 9

    # Instantiate model (no from_pretrained)
    model = VLNBert(config=vis_config)

    # Load weights manually
    print(f"🔹 Loading weights from {model_path} ...")
    state_dict = torch.load(model_path, map_location='cpu')

    # 有的 checkpoint 是 DataParallel 格式，需去掉 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"✅ Model loaded successfully!")
    print(f"   Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model
