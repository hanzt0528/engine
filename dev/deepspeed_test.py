import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam,FusedAdam
import deepspeed.comm as dist

import deepspeed

if args.local_rank==-1:
    