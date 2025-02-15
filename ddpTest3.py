import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import datetime

# 1. Custom ProcessGroup Implementation

class CustomProcessGroup(ProcessGroup):
    """
    A custom ProcessGroup implementation.
    """

    def __init__(self, rank, size, group_name, timeout=datetime.timedelta(seconds=30)):
        super().__init__(rank, size, timeout=timeout)
        self.rank = rank
        self.size = size
        self.group_name = group_name
        print(f"CustomProcessGroup initialized: rank={rank}, size={size}, group_name={group_name}")

    # ... (rest of the methods: allreduce, broadcast, allgather, barrier - no changes) ...
    def allreduce(self, tensors, op=dist.ReduceOp.SUM, async_op=False):
        """
        All-reduces the tensor across the group.

        Args:
            tensors (List[torch.Tensor]): List containing a single tensor to all-reduce.
            op (dist.ReduceOp, optional): Reduction operation (SUM, MIN, MAX, PRODUCT). Defaults to SUM.
            async_op (bool, optional):  Whether the operation is asynchronous.  Not supported in this basic implementation.

        Returns:
            torch.futures.Future: A future object (completed immediately in this sync implementation).
        """
        if async_op:
            raise NotImplementedError("Asynchronous operations not supported in CustomProcessGroup")
        if len(tensors) != 1:
            raise ValueError("CustomProcessGroup.allreduce only supports a single tensor.")

        tensor = tensors[0]

        # Check if the tensor is on CPU, if not, move it to CPU
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        reduced_tensor = tensor.clone()

        for i in range(self.size):
            if i != self.rank:
                # Simulate receiving from other ranks (blocking)
                other_tensor = torch.empty_like(tensor) # Allocate space.
                if self.rank < i: #Simple send/recv to avoid deadlock
                    torch.distributed.send(tensor, i)
                    torch.distributed.recv(other_tensor, i)
                else:
                    torch.distributed.recv(other_tensor, i)
                    torch.distributed.send(tensor, i)

                # Apply the reduction operation
                if op == dist.ReduceOp.SUM:
                    reduced_tensor += other_tensor
                elif op == dist.ReduceOp.MIN:
                    reduced_tensor = torch.minimum(reduced_tensor, other_tensor)
                elif op == dist.ReduceOp.MAX:
                    reduced_tensor = torch.maximum(reduced_tensor, other_tensor)
                elif op == dist.ReduceOp.PRODUCT:
                    reduced_tensor *= other_tensor
                else:
                    raise ValueError(f"Unsupported reduction operation: {op}")

        # Update the original tensor
        tensor.copy_(reduced_tensor)
        tensors[0] = tensor # Modify the input list in place (required by ProcessGroup API).
        future = torch.futures.Future()
        future.set_result(None) # Mark as immediately completed (synchronous)
        return future



    def broadcast(self, tensors, src, async_op=False):
        """
        Broadcasts the tensor from the source rank to all other ranks.

        Args:
            tensors (List[torch.Tensor]): List containing a single tensor to broadcast.
            src (int): Source rank.
            async_op (bool, optional): Whether the operation is asynchronous.

        Returns:
            torch.futures.Future: A future object.
        """

        if async_op:
            raise NotImplementedError("Asynchronous operations not supported")
        if len(tensors) != 1:
            raise ValueError("CustomProcessGroup.broadcast only supports a single tensor.")

        tensor = tensors[0]

        # Check if the tensor is on CPU, if not, move it to CPU
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()

        if self.rank == src:
            # Source rank: Send the tensor to all other ranks.
            for i in range(self.size):
                if i != src:
                    torch.distributed.send(tensor, i)
        else:
            # Other ranks: Receive the tensor from the source rank.
            torch.distributed.recv(tensor, src) # tensor already allocated

        tensors[0] = tensor # Modify in-place
        future = torch.futures.Future()
        future.set_result(None)
        return future



    def allgather(self, output_tensors, input_tensors, async_op=False):
        """
        All-gathers tensors from all ranks and stores them in output_tensors.

        Args:
            output_tensors (List[List[torch.Tensor]]):  List of lists of tensors to store results.
                output_tensors[i] will contain the gathered tensor from rank i.
            input_tensors (List[torch.Tensor]):  List containing a single tensor to gather.
            async_op (bool, optional): Whether the operation is asynchronous.

        Returns:
            torch.futures.Future: A future object.
        """

        if async_op:
            raise NotImplementedError("Asynchronous operations not supported")
        if len(input_tensors) != 1:
            raise ValueError("CustomProcessGroup.allgather only supports a single input tensor.")
        if len(output_tensors) != self.size:
             raise ValueError(f"output_tensors should have length {self.size}")

        input_tensor = input_tensors[0]

        # Check if the input tensor is on CPU, if not, move it to CPU
        if input_tensor.device != torch.device('cpu'):
            input_tensor = input_tensor.cpu()

        # Send and receive tensors to/from all other ranks.
        for i in range(self.size):
             if len(output_tensors[i]) != 1:
                raise ValueError("Each list in output_tensors should contain exactly one tensor.")

             # Check if the tensor is on CPU, if not, move it to CPU
             if output_tensors[i][0].device != torch.device('cpu'):
                output_tensors[i][0] = output_tensors[i][0].cpu()

             if i == self.rank:
                output_tensors[i][0].copy_(input_tensor)  # Copy local tensor

             if self.rank < i: # Simple send/recv to avoid deadlock
                torch.distributed.send(input_tensor, i)
                torch.distributed.recv(output_tensors[i][0],i)
             elif self.rank > i:
                torch.distributed.recv(output_tensors[i][0],i)
                torch.distributed.send(input_tensor, i)


        future = torch.futures.Future()
        future.set_result(None)
        return future

    def barrier(self, async_op=False):
        """
        Simple barrier implementation.  All ranks send a tensor to rank 0,
        and rank 0 receives from all.  Then rank 0 broadcasts to all others.
        """
        if async_op:
            raise NotImplementedError("Asynchronous barrier not supported")

        tensor = torch.tensor([self.rank], dtype=torch.int)

        if self.rank == 0:
            for i in range(1, self.size):
                recv_tensor = torch.empty_like(tensor)
                torch.distributed.recv(recv_tensor, i)
        else:
            torch.distributed.send(tensor, 0)

        self.broadcast([tensor], src=0) # Use the broadcast we already defined.

        future = torch.futures.Future()
        future.set_result(None)
        return future

# Factory function to create the CustomProcessGroup (CORRECTED)
def custom_process_group_factory(*args, **kwargs):
    rank = args[1]
    size = args[2]
    group_name = f"custom_group_{rank}_{size}"
    return CustomProcessGroup(rank, size, group_name, timeout=kwargs.get('timeout', datetime.timedelta(seconds=30)))

# 2. Backend Registration (using the factory function)
dist.Backend.register_backend("custom_cpu", custom_process_group_factory)

# 3. Driving Program (no changes needed here)
def init_process(rank, size, backend):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=size)

def cleanup():
    dist.destroy_process_group()

def triton_kernel(a, b, out):
    """A simple Triton kernel (element-wise addition)."""
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(
        a_ptr, b_ptr, output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        output = a + b
        tl.store(output_ptr + offsets, output, mask=mask)

    N = a.numel()
    assert a.shape == b.shape == out.shape
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](a, b, out, N, BLOCK_SIZE=1024)
    return out

def run_example(rank, size):
    init_process(rank, size, backend="custom_cpu")
    print(f"Rank {rank} initialized.")

    # Create tensors
    tensor_size = 1024 * 10
    if rank == 0:
       tensor_a = torch.randn(tensor_size, device="cpu")
       tensor_b = torch.randn(tensor_size, device="cpu")
    else:
       tensor_a = torch.zeros(tensor_size, device="cpu")
       tensor_b = torch.zeros(tensor_size, device="cpu")
    tensor_out = torch.empty_like(tensor_a)

    # Broadcast tensors from rank 0
    dist.broadcast(tensor_a, src=0)
    dist.broadcast(tensor_b, src=0)

    # Compile the Triton kernel with torch.compile
    compiled_kernel = torch.compile(triton_kernel, backend="inductor")

    # Run the compiled kernel
    compiled_kernel(tensor_a, tensor_b, tensor_out)


    # All-gather the results (optional, for verification)
    gathered_tensors = [torch.empty_like(tensor_out) for _ in range(size)]
    dist.all_gather(gathered_tensors, tensor_out)

    # All-reduce a different tensor to demonstrate allreduce
    reduce_tensor = torch.tensor([float(rank + 1)], device="cpu")  # Different values on each rank
    dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)

    dist.barrier() # Make sure all processes finish before cleanup
    if rank == 0:
        # Verify the all-gathered results.  Each gathered tensor should be
        # equal to tensor_out (which is tensor_a + tensor_b).
        for i, gathered_tensor in enumerate(gathered_tensors):
             assert torch.allclose(gathered_tensor, tensor_a + tensor_b), f"Rank {rank}: Allgather verification failed at rank {i}."
        # Verify the all-reduce result.  The sum should be (size * (size + 1)) / 2.
        expected_sum = (size * (size + 1)) / 2
        assert torch.allclose(reduce_tensor, torch.tensor([expected_sum])), f"Allreduce verification failed. Expected {expected_sum}, got {reduce_tensor}"
        print("All verifications passed!")


    cleanup()

if __name__ == "__main__":
    size = 4  # Number of processes
    # Spawn processes
    torch.multiprocessing.spawn(run_example,
        args=(size,),
        nprocs=size,
        join=True)
