import os
import time
import torch
import psutil
import numpy as np
from thop import profile

from map_nav_src_fullmodule.r2r.parser import parse_args
from map_nav_src_fullmodule.models.model import VLNBert


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params / 1e6:.3f} M')
    return total_params


def count_module_parameters(model):
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = params / 1e6

    print("\n======= Parameter Breakdown (M) =======")
    total = 0
    for k, v in module_params.items():
        print(f"{k:<20}: {v:.3f} M")
        total += v
    print(f"{'-'*30}\nTotal (Check): {total:.3f} M")
    print("=======================================\n")
    return module_params


def measure_module_stats(model, mode, inputs, warmup=5, repeat=20, device="cuda:1"):
    """测量延迟 + 吞吐量 + GPU/CPU内存"""
    model.eval()
    torch.cuda.synchronize(device)
    process = psutil.Process(os.getpid())

    # Warm-up
    for _ in range(warmup):
        with torch.no_grad():
            model(mode, inputs)
    torch.cuda.synchronize(device)

    # 重置GPU显存峰值
    torch.cuda.reset_peak_memory_stats(device)
    ram_before = process.memory_info().rss / 1024**2

    start = time.time()
    for _ in range(repeat):
        with torch.no_grad():
            model(mode, inputs)
    torch.cuda.synchronize(device)
    end = time.time()

    ram_after = process.memory_info().rss / 1024**2
    peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**2

    latency = (end - start) / repeat * 1000  # ms
    throughput = 1000 / latency  # 每秒可处理样本数

    ram_used = ram_after - ram_before

    return latency, throughput, peak_gpu, ram_used


def GFLOPs_latency_memory(vln_bert, bs, txt_lens=44, h_dim=768, device="cuda:1"):
    vln_bert.eval()
    print('\n======= GFLOPs, Latency & Memory Evaluation =======')
    results = {}

    # Helper: 自动转到指定设备
    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, list):
            return [to_device(i) for i in x]
        elif isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        return x

    # ---------- Language ----------
    txt_ids = torch.randint(0, 2000, size=(bs, txt_lens), device=device)
    txt_masks = torch.ones(bs, txt_lens, dtype=torch.bool, device=device)
    lan_input = {'txt_ids': txt_ids, 'txt_masks': txt_masks}

    lan_macs, _ = profile(vln_bert, inputs=('language', lan_input), verbose=False)
    lan_gflops = lan_macs * 2 / 1e9
    lan_latency, lan_thpt, lan_gpu, lan_ram = measure_module_stats(vln_bert, 'language', lan_input, device=device)
    results['Language'] = (lan_gflops, lan_latency, lan_thpt, lan_gpu, lan_ram)

    # ---------- Panorama ----------
    pano_input = {
        'view_img_fts': torch.rand(bs, 37, h_dim, device=device),
        'view_text_fts': torch.rand(bs, 37, h_dim, device=device),
        'loc_fts': torch.rand(bs, 37, 7, device=device),
        'nav_types': torch.randint(0, 2, (bs, 37), device=device).long(),
        'view_lens': torch.ones(bs, device=device).long() * 37,
        'obj_lens': torch.ones(bs, device=device).long() * 10,
    }
    pano_input = to_device(pano_input)

    pan_macs, _ = profile(vln_bert, inputs=('panorama', pano_input), verbose=False)
    pan_gflops = pan_macs * 2 / 1e9
    pan_latency, pan_thpt, pan_gpu, pan_ram = measure_module_stats(vln_bert, 'panorama', pano_input, device=device)
    results['Panorama'] = (pan_gflops, pan_latency, pan_thpt, pan_gpu, pan_ram)

    # ---------- Navigation ----------
    grid_seq_len = 16 * 16
    nav_input = {
        'txt_embeds': torch.rand(bs, 44, h_dim, device=device),
        'txt_masks': torch.ones(bs, 44, dtype=torch.bool, device=device),
        'gmap_img_embeds': torch.rand(bs, 5, h_dim, device=device),
        'gmap_step_ids': torch.randint(0, 2, (bs, 5), device=device).long(),
        'gmap_pos_fts': torch.rand(bs, 5, 7, device=device),
        'gmap_masks': torch.ones(bs, 5, dtype=torch.bool, device=device),
        'gmap_pair_dists': torch.rand(bs, 5, 5, device=device),
        'gmap_visited_masks': torch.ones(bs, 5, dtype=torch.bool, device=device),
        'gmap_vpids': [['vp%d' % i for i in range(5)] for _ in range(bs)],
        'vp_img_embeds': torch.rand(bs, 5, h_dim, device=device),
        'vp_pos_fts': torch.rand(bs, 5, 14, device=device),
        'vp_masks': torch.ones(bs, 5, dtype=torch.bool, device=device),
        'vp_nav_masks': torch.ones(bs, 5, dtype=torch.bool, device=device),
        'vp_cand_vpids': [['vp_cand_%d' % j for j in range(5)] for _ in range(bs)],
        'grid_spatial_text': [torch.rand(5, 12, 768, device=device) for _ in range(bs)],
        'grid_fts': [torch.rand((5 * 12 * 64, 768), device=device) for _ in range(bs)],
        'grid_map': [torch.rand((5 * 12 * 64,), device=device) for _ in range(bs)],
        'gridmap_pos_fts': torch.rand(bs, grid_seq_len, 5, device=device)
    }
    nav_input = to_device(nav_input)

    nav_macs, _ = profile(vln_bert, inputs=('navigation', nav_input), verbose=False)
    nav_gflops = nav_macs * 2 / 1e9
    nav_latency, nav_thpt, nav_gpu, nav_ram = measure_module_stats(vln_bert, 'navigation', nav_input, device=device)
    results['Navigation'] = (nav_gflops, nav_latency, nav_thpt, nav_gpu, nav_ram)

    # ---------- 输出汇总 ----------
    print("\n====================== Summary ======================")
    print(f"{'Module':<12} | {'GFLOPs':<8} | {'Latency(ms)':<12} | {'Throughput':<10} | {'Peak GPU(MB)':<12} | {'RAM(MB)':<8}")
    print("-" * 80)

    total_gflops, total_latency, total_gpu, total_ram = 0, 0, 0, 0
    for k, (gflops, latency, thpt, gpu, ram) in results.items():
        print(f"{k:<12} | {gflops:<8.3f} | {latency:<12.3f} | {thpt:<10.2f} | {gpu:<12.1f} | {ram:<8.1f}")
        total_gflops += gflops
        total_latency += latency
        total_gpu = max(total_gpu, gpu)
        total_ram += ram

    print("-" * 80)
    print(f"{'Total':<12} | {total_gflops:<8.3f} | {total_latency:<12.3f} | {'-':<10} | {total_gpu:<12.1f} | {total_ram:<8.1f}")
    print("=====================================================\n")

    return results


if __name__ == '__main__':
    args = parse_args()

    # 指定 GPU 1
    device = torch.device("cuda:1")
    vln_bert = VLNBert(args).to(device)

    # 总参数量
    count_parameters(vln_bert)

    # 模块参数量
    count_module_parameters(vln_bert)

    # GFLOPs + 延迟 + 吞吐量 + GPU/CPU 内存
    bs = 8
    GFLOPs_latency_memory(vln_bert, bs, device=device)
