import torch


def get_tokenizer(args):
    """
    获取分词器。

    参数:
        args: 参数对象，包含配置信息。
            - args.tokenizer: 指定使用的分词器类型，支持 'xlm' 或其他（默认为 BERT）。

    返回:
        tokenizer: 分词器对象，用于处理文本数据。
    """
    from transformers import AutoTokenizer  # 导入 Hugging Face 的 AutoTokenizer

    # 根据参数选择预训练模型的配置名称
    if args.tokenizer == 'xlm':  # 如果指定使用 XLM-RoBERTa 分词器
        cfg_name = 'xlm-roberta-base'  # 使用 XLM-RoBERTa 的基础模型
    else:  # 默认使用 BERT 分词器
        cfg_name = '/home/files/A/zhanghuaxiang3/GridMM/datasets/bert-base'  # 指定 BERT 模型的路径

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg_name, local_files_only=True)
    # 使用 `from_pretrained` 方法加载指定的预训练模型分词器，`local_files_only=True` 表示只从本地加载文件

    return tokenizer  # 返回分词器对象

def get_vlnbert_models(args, config=None):
    from transformers import PretrainedConfig
    from map_nav_src_fullmodule.models.vilmodel import GlocalTextPathNavCMT
    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v

    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = '/home/files/A/zhanghuaxiang3/GridMM/datasets/bert-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name, local_files_only=True)

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False

    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights, ignore_mismatched_sizes=True)

    return visual_model
