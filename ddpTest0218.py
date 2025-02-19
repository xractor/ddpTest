import torch
import torch.distributed as dist
from torch.distributed import Backend, ProcessGroup
from torch.distributed._tensor import DeviceMesh, DTensor  # for DTensor support
import os

# Placeholder for custom communication algorithms (replace with your hardware-specific logic)
def custom_all_reduce(tensor, op):
    # VERY simple CPU-based all-reduce (for demonstration)
    dist.all_reduce(tensor, op=op)  # Delegate to default backend in this example

def custom_broadcast(tensor, src):
    # VERY simple CPU-based broadcast (for demonstration)
    dist.broadcast(tensor, src=src)

def custom_all_gather(tensor_list, tensor):
    # VERY simple CPU-based all-gather (for demonstration)
    dist.all_gather(tensor_list, tensor)

class CustomBackend(Backend):
    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        # Any backend-specific initialization, if needed

    def all_reduce(self, tensor, op, group=ProcessGroup.WORLD, async_op=False):
        custom_all_reduce(tensor, op) # using default all reduce in this demo
        if async_op:
            return torch.futures.Future()  # Placeholder for asynchronous operation
        return None

    def broadcast(self, tensor, src, group=ProcessGroup.WORLD, async_op=False):
        custom_broadcast(tensor, src) # using default broadcast in this demo
        if async_op:
            return torch.futures.Future()  # Placeholder for asynchronous operation
        return None
    def all_gather(self, tensor_list, tensor, group=ProcessGroup.WORLD, async_op=False):

        custom_all_gather(tensor_list, tensor) # using default all_gather in this demo
        if async_op:
            return torch.futures.Future()  # Placeholder for asynchronous operation
        return None

    def _barrier(self, group=ProcessGroup.WORLD, async_op=False): # barrier is not abstractmethod since 2.1, override it for safety
        dist.barrier(group=group)
        if async_op:
            return torch.futures.Future()
        return None

    def is_initialized(self):
        return True # always return true in this example.

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size
    
# --- Triton Kernel Example (Illustrative, not part of the backend) ---
# This is how a user might define a Triton kernel.  Crucially, the output is a standard PyTorch tensor.
# (This example just adds 1 to each element, for demonstration)

import triton
import triton.language as tl

@triton.jit
def add_one_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 1
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add_one(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_one_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output


# --- Verification Driver Program ---

def run_test(rank, world_size, backend_name):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if backend_name == "custom":
        dist.init_process_group(backend=CustomBackend, rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend=backend_name, rank=rank, world_size=world_size)
        

    # 1. Test with Basic Tensors (All-Reduce)
    tensor = torch.ones(10) * (rank + 1)
    print(f"Rank {rank}: Initial tensor: {tensor}")

    if backend_name == "custom":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM) # use defaut backend all_reduce
    else:
         dist.all_reduce(tensor, op=dist.ReduceOp.SUM) # use defaut backend all_reduce
    
    expected_sum = (world_size * (world_size + 1)) / 2 * 10
    assert torch.allclose(tensor, torch.ones(10) * expected_sum), f"Rank {rank}: All-reduce failed. Got {tensor}, expected {expected_sum}"
    print(f"Rank {rank}: All-reduce successful. Result: {tensor}")

    # 2. Test with Triton (All-Reduce)
    tensor_triton = torch.ones(10) * (rank + 1)
    tensor_triton = triton_add_one(tensor_triton)  # Apply Triton kernel
    print(f"Rank {rank}: Triton-processed tensor: {tensor_triton}")

    if backend_name == "custom":
        dist.all_reduce(tensor_triton, op=dist.ReduceOp.SUM)  # Custom backend
    else:
        dist.all_reduce(tensor_triton, op=dist.ReduceOp.SUM)
    expected_sum_triton = (world_size * (world_size + 1)) / 2 * 10 + world_size * 10 # added one by triton kernel
    assert torch.allclose(tensor_triton, torch.ones(10) * expected_sum_triton), f"Rank {rank}: Triton All-reduce failed. Got {tensor_triton}, expected {expected_sum_triton}"
    print(f"Rank {rank}: Triton All-reduce successful. Result: {tensor_triton}")

    # 3. Test with DTensor (All-Reduce)
    if rank == 0:
        device_mesh = DeviceMesh("cpu", torch.arange(0, world_size))
    else:
         device_mesh = DeviceMesh("cpu", torch.arange(0, world_size)) # for custom backend.
    
    shard_spec = [dist.Shard(0)]  # Shard along the first dimension
    tensor_dtensor = torch.ones(world_size, 10) * (rank + 1)  # Create a larger tensor on rank 0
    dtensor = DTensor.from_local(tensor_dtensor[rank], device_mesh, shard_spec)
    print(f"Rank {rank}: DTensor local shard: {dtensor.to_local()}")
    
    if backend_name == "custom":
        dtensor_reduced = dtensor.all_reduce(op=torch.distributed.ReduceOp.SUM)
    else:
       dtensor_reduced = dtensor.all_reduce()

    expected_sum_dtensor = (world_size * (world_size + 1)) / 2 * 10
    assert torch.allclose(dtensor_reduced.to_local(), torch.ones(1, 10) * expected_sum_dtensor), f"Rank {rank}: DTensor All-reduce failed. Got {dtensor_reduced.to_local()}, expected {expected_sum_dtensor}"
    print(f"Rank {rank}: DTensor All-reduce successful. Result: {dtensor_reduced.to_local()}")

    # 4. Test all gather
    tensor_all_gather = torch.ones(10) * (rank + 1)
    tensor_list = [torch.zeros(10) for _ in range(world_size)]

    if backend_name == "custom":
        dist.all_gather(tensor_list, tensor_all_gather)
    else:
        dist.all_gather(tensor_list, tensor_all_gather)

    for i in range(world_size):
        assert torch.allclose(tensor_list[i], torch.ones(10) * (i + 1)), f"AllGather failed, rank {rank} element {i}"

    # 5. Test broadcast
    tensor_broadcast = torch.ones(10) * (rank + 1)
    if backend_name == "custom":
       dist.broadcast(tensor_broadcast, src = 0)
    else:
        dist.broadcast(tensor_broadcast, src=0)

    if rank != 0:
        assert torch.allclose(tensor_broadcast,  torch.ones(10)), f"Broadcast failed at rank {rank}."
    print(f"Rank {rank}: Broadcast successful. Result: {tensor_broadcast}")

    dist.destroy_process_group()
    print(f"Rank {rank}: Test completed.")

if __name__ == "__main__":
    world_size = 2  # Set the number of processes
    torch.multiprocessing.spawn(run_test, args=(world_size, "custom"), nprocs=world_size, join=True)
    torch.multiprocessing.spawn(run_test, args=(world_size, "gloo"), nprocs=world_size, join=True)
