import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.tensor.api import Shard
import triton  # We'll import triton but use it minimally
import os

# --- Custom Process Group (Simplified) ---

class MyProcessGroup(ProcessGroup):
    def __init__(self, rank, size):
        super().__init__(rank, size)
        # In a real implementation, you'd initialize your custom hardware connection here.

    def allreduce(self, tensor, op=dist.ReduceOp.SUM):
        # Simplified allreduce using gloo.
        dist.all_reduce(tensor, op, group=dist.group.WORLD)  # Delegate to gloo
        return [dist.Work()]  # Return a list of Work objects, even if it's trivial

    def broadcast(self, tensor, src=0):
        # Simplified broadcast using gloo.
        dist.broadcast(tensor, src, group=dist.group.WORLD)
        return [dist.Work()]

    def allgather(self, output_tensor_list, input_tensor):
        # Simplified allgather using gloo
        dist.all_gather(output_tensor_list, input_tensor, group=dist.group.WORLD)
        return [dist.Work()]
    
    def barrier(self):
        dist.barrier(group=dist.group.WORLD)
        return [dist.Work()]
    
# --- Registration Function ---

def _my_pg_init(store, rank, size, timeout):
  return MyProcessGroup(rank, size)

dist.Backend.register_backend("my_custom_backend", _my_pg_init)


# --- Triton Kernel (Example) ---
# This is a very simple Triton kernel, just to show interaction.
@triton.jit
def my_kernel(x_ptr, out_ptr, n_elements):
    pid = triton.program_id(axis=0)
    block_size = 1024
    offset = pid * block_size
    mask = offset + triton.arange(0, block_size) < n_elements
    x = triton.load(x_ptr + offset, mask=mask)
    y = x * 2  # Some simple computation
    triton.store(out_ptr + offset, y, mask=mask)

# --- Driving Program ---

def run_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize with our custom backend.
    dist.init_process_group(backend="my_custom_backend", rank=rank, world_size=world_size)
    # Create a DeviceMesh for DTensor (if using DTensor)
    device_mesh = DeviceMesh("cpu", list(range(world_size)))

    # --- Test with DTensor ---
    tensor_size = 1024
    # Create a local tensor
    local_tensor = torch.randn(tensor_size // world_size)
    # Create a DTensor sharded across the mesh.
    dtensor = DTensor.from_local(local_tensor, device_mesh, [Shard(0)])
    # All-reduce the DTensor.
    my_pg = dist.new_group(backend="my_custom_backend")
    my_pg.allreduce(dtensor)
    # Check the result (each shard should have the sum).
    expected_sum = torch.sum(torch.randn(tensor_size // world_size) * world_size)
    
    if rank == 0:
        print(f"DTensor allreduce result (first element): {dtensor.to_local()[0]}")
    
    assert torch.allclose(dtensor.to_local()[0], expected_sum), "DTensor allreduce failed!"

    # --- Test with Regular Tensor (and Triton) ---

    # Create a tensor.
    x = torch.randn(tensor_size)
    # Run a Triton kernel.
    out = torch.empty_like(x)
    my_kernel[(tensor_size + 1023) // 1024](x, out, tensor_size)

    # All-reduce the result from Triton.
    my_pg.allreduce(out)
    if rank == 0:
        print(f"Triton + allreduce result (first element): {out[0]}")

    # --- Test Broadcast ---
    if rank == 0:
      bcast_tensor = torch.tensor([42.0], dtype=torch.float32)
    else:
      bcast_tensor = torch.tensor([0.0], dtype=torch.float32)
    
    my_pg.broadcast(bcast_tensor, src=0)
    if rank == 1:
        print(f"Broadcast result on rank 1: {bcast_tensor[0]}")
    assert torch.allclose(bcast_tensor, torch.tensor([42.0])), "Broadcast failed!"


    # --- Test Allgather ---
    tensor_to_gather = torch.full((1,), rank, dtype=torch.float32)
    gathered_tensors = [torch.empty_like(tensor_to_gather) for _ in range(world_size)]
    my_pg.allgather(gathered_tensors, tensor_to_gather)
    if rank == 0:
        print("Allgather results on rank 0:", [t.item() for t in gathered_tensors])
    assert torch.allclose(torch.stack(gathered_tensors), torch.arange(world_size, dtype=torch.float32)), "Allgather failed!"


    dist.destroy_process_group()
    print(f"Rank {rank} finished successfully.")

if __name__ == "__main__":
    world_size = 2  # Run with 2 processes.
    torch.multiprocessing.spawn(run_test, args=(world_size,), nprocs=world_size)
