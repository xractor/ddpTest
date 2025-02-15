import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# 1. Custom ProcessGroup Implementation

class CustomProcessGroup(ProcessGroup):
    """
    A custom ProcessGroup implementation for simplified collective operations.
    This implementation focuses on CPU operations for easy testing and uses
    basic torch tensor operations for communication.  It's *not* optimized for
    performance and serves primarily as an educational example.

    Args:
        rank (int): Rank of the current process.
        size (int): Total number of processes in the group.
    """
    def __init__(self, rank, size):
        super().__init__(rank, size)
        self.rank = rank
        self.size = size
        self._is_init = True  # Explicit initialization flag
        print(f"CustomProcessGroup initialized: rank={rank}, size={size}")

    def allreduce(self, tensor, op=dist.ReduceOp.SUM):
        """
        Performs an all-reduce operation on the given tensor.

        Args:
            tensor (torch.Tensor): The tensor to be reduced.  Must be on CPU.
            op (dist.ReduceOp): The reduction operation (default: SUM).
                Currently, only SUM is supported.

        Returns:
            None (in-place operation)
        """
        if not self._is_init:
            raise RuntimeError("CustomProcessGroup not initialized.")
        if not tensor.is_cpu:
            raise ValueError("CustomProcessGroup only supports CPU tensors.")
        if op != dist.ReduceOp.SUM:
            raise NotImplementedError("Only SUM operation is currently supported.")

        # Gather all tensors to rank 0
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.size)] if self.rank == 0 else None
        dist.gather(tensor, gathered_tensors, dst=0, group=self)

        if self.rank == 0:
            # Perform the reduction on rank 0
            reduced_tensor = torch.sum(torch.stack(gathered_tensors), dim=0)
            # Scatter the result to all ranks
            dist.scatter(reduced_tensor, [reduced_tensor for _ in range(self.size)], src=0, group=self)
        else:
            # Other ranks receive the scattered result
            dist.scatter(tensor, [], src=0, group=self) # tensor is overwritten

    def broadcast(self, tensor, src):
        """
        Broadcasts a tensor from the source rank to all other ranks.

        Args:
            tensor (torch.Tensor): The tensor to be broadcast. Must be on CPU.
            src (int): The rank of the source process.

        Returns:
            None (in-place operation)
        """
        if not self._is_init:
            raise RuntimeError("CustomProcessGroup not initialized.")
        if not tensor.is_cpu:
            raise ValueError("CustomProcessGroup only supports CPU tensors.")

        dist.broadcast(tensor, src=src, group=self) # Use built in broadcast

    def allgather(self, output_tensor_list, input_tensor):
        """
        Gathers tensors from all ranks into a list of tensors.

        Args:
            output_tensor_list (list[torch.Tensor]): A list of tensors to
                receive the gathered data.  Must be of length `size` and each
                tensor must have the same shape as `input_tensor`. Must be on CPU.
            input_tensor (torch.Tensor): The tensor to be gathered from this rank.
                Must be on CPU.

        Returns:
            None (in-place operation)
        """
        if not self._is_init:
            raise RuntimeError("CustomProcessGroup not initialized.")
        if not all(t.is_cpu for t in output_tensor_list) or not input_tensor.is_cpu:
            raise ValueError("CustomProcessGroup only supports CPU tensors.")
        if len(output_tensor_list) != self.size:
            raise ValueError("output_tensor_list must have length equal to the process group size.")
        if not all(t.shape == input_tensor.shape for t in output_tensor_list):
            raise ValueError("All tensors in output_tensor_list must have the same shape as input_tensor.")

        dist.all_gather(output_tensor_list, input_tensor, group=self) # Use built in all_gather

    def barrier(self, opts=None):
        """
        Placeholder implementation of barrier since we use built-in all_gather
        and broadcast.
        """
        if not self._is_init:
          raise RuntimeError("CustomProcessGroup not initialized.")
        
        # Simple barrier using all-reduce (overkill, but works)
        temp_tensor = torch.tensor([1.0], device="cpu")
        self.allreduce(temp_tensor)

    def send(self, tensor, dstRank, tag=0):
        if not self._is_init:
            raise RuntimeError("CustomProcessGroup not initialized.")
        dist.send(tensor, dstRank, group=self) # use built in functions
        

    def recv(self, tensor, srcRank, tag=0):
        if not self._is_init:
            raise RuntimeError("CustomProcessGroup not initialized.")

        dist.recv(tensor, srcRank, group=self) # use built in functions


# 2. Register Custom Backend

def init_custom_pg(rank, size):
    """Initializes the custom process group."""
    print(f"Initializing CustomProcessGroup: rank={rank}, size={size}")
    return CustomProcessGroup(rank, size)


dist.register_backend("custom", init_custom_pg)


# 3. Triton Kernel (Example)

import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple element-wise addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# 4. Driving Program

def run_custom_distributed(rank, world_size, n):
    """
    Runs a simple distributed program using the custom process group.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        n (int): Size of the tensors.
    """

    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Choose an available port
    dist.init_process_group("custom", rank=rank, world_size=world_size)


    # Create tensors
    x = torch.randn(n, device="cpu")
    y = torch.randn(n, device="cpu")
    out = torch.zeros(n, device="cpu")
    
    # Compile and run the Triton kernel
    compiled_add = torch.compile(add_kernel, backend="inductor")  # Or "torchscript"
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    compiled_add[grid](x, y, out, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize() # Not really needed for CPU execution, just keep the api for consistency.
    print(f"Rank {rank}: initial out = {out}")

    # All-reduce the output using the custom process group
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=None) # use default process group
    print(f"Rank {rank}: all-reduced out = {out}")

    # Broadcast test
    if rank == 0:
      to_broadcast = torch.tensor([42.0 + rank], device="cpu")
    else:
      to_broadcast = torch.tensor([0.0], device="cpu")  # Initialize on other ranks
      
    dist.broadcast(to_broadcast, src=0)
    print(f"Rank {rank}: broadcasted value = {to_broadcast}")
    
    # Allgather test
    tensor_to_gather = torch.ones(2, device='cpu') * (rank + 1)  # different values on each rank
    gathered_tensors = [torch.zeros(2, device='cpu') for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor_to_gather)
    print(f"Rank {rank}: gathered tensors = {gathered_tensors}")


    # Clean up
    dist.destroy_process_group()



if __name__ == "__main__":
    n = 1024 * 16
    world_size = 4  # Example: Run with 4 processes
    torch.multiprocessing.spawn(run_custom_distributed, args=(world_size, n), nprocs=world_size)
