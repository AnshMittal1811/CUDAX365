import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    x = torch.ones(1024, device="cuda") * (rank + 1)
    dist.all_reduce(x)
    if rank == 0:
        print("allreduce sum", float(x[0]))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
