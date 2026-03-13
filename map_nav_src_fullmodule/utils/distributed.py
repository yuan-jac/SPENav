"""
Distributed tools
"""
import os
import pickle
from pathlib import Path

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def load_init_param(opts):
    """
    Prepare parameters for distributed initialization
    """
    # sync file
    if opts.output_dir != "":
        sync_dir = Path(opts.output_dir).resolve()
        sync_dir.mkdir(parents=True, exist_ok=True)
        sync_file = f"{sync_dir}/.torch_distributed_sync"
    else:
        raise RuntimeError("Can't find any sync dir")

    # World Size
    if opts.world_size != -1:
        world_size = opts.world_size
    elif "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        raise RuntimeError("Can't find any world size.")

    # Rank
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif opts.node_rank != -1:
        # Manually set node_rank + local_rank
        if opts.local_rank != -1:
            rank = opts.node_rank * torch.cuda.device_count() + opts.local_rank
        else:
            raise RuntimeError("Can't find local rank")
    else:
        raise RuntimeError("Can't find rank.")

    return {
        "backend": "nccl",
        "init_method": f"file://{sync_file}",
        "world_size": world_size,
        "rank": rank,
    }


def init_distributed(opts):
    init_param = load_init_param(opts)

    print(f"[Distributed Init] rank {init_param['rank']}, world_size {init_param['world_size']}")
    dist.init_process_group(**init_param)

    # set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if "LOCAL_RANK" in os.environ else opts.local_rank
    torch.cuda.set_device(local_rank)

    return init_param["rank"]


def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or get_rank() == 0


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialize to tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # gather sizes
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # gather tensors
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))

    if local_size != max_size:
        padding = torch.empty((max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        # sort keys
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)
    return outs
