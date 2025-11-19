import os
import argparse
from sonic import Sonic
import logging
import sys
import torch
import torch.distributed as dist
from src.distributed.util import init_distributed_group



def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)



parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('audio_path')
parser.add_argument('output_path')
parser.add_argument('--dynamic_scale', type=float, default=1.0)
parser.add_argument('--crop', action='store_true')
parser.add_argument('--seed', type=int, default=None)

parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")

args = parser.parse_args()

rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
local_rank = int(os.getenv("LOCAL_RANK", 0))
device = local_rank
_init_logging(rank)

if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size)
else:
    assert not (
        args.dit_fsdp
    ), f"dit_fsdp is not supported in non-distributed environments."
    assert not (
        args.ulysses_size > 1
    ), f"sequence parallel are not supported in non-distributed environments."

if args.ulysses_size > 1:
    assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
    init_distributed_group()

if dist.is_initialized():
    seed = [args.seed] if rank == 0 else [None]
    dist.broadcast_object_list(seed, src=0)
    args.seed = seed[0]

pipe = Sonic(
    device_id=device,
    dit_fsdp=args.dit_fsdp,
    use_sp=(args.ulysses_size > 1)
    )


face_info = pipe.preprocess(args.image_path, expand_ratio=0.5)
print(face_info)
if face_info['face_num'] >= 0:
    if args.crop:
        crop_image_path = args.image_path + '.crop.png'
        if rank==0:
            pipe.crop_image(args.image_path, crop_image_path, face_info['crop_bbox'])
            args.image_path = crop_image_path
        if dist.is_initialized():
            image_path = [args.image_path] if rank == 0 else [None]
            dist.broadcast_object_list(image_path, src=0)
            args.image_path = image_path[0]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pipe.process(args.image_path, args.audio_path, args.output_path, min_resolution=512, inference_steps=25, dynamic_scale=args.dynamic_scale, seed=args.seed)
