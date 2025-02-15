import torch
import torch.distributed as dist

###############################################################################
# 1. Define a custom Work handle that mimics an asynchronous operation.
###############################################################################
class CustomWork(dist.Work):
    def __init__(self, tensor):
        self._tensor = tensor

    def wait(self):
        # In a real async operation, wait for completion.
        return self._tensor

###############################################################################
# 2. Define a custom ProcessGroup by subclassing torch.distributed.ProcessGroup.
###############################################################################
class CustomProcessGroup(dist.ProcessGroup):
    def __init__(self, store, rank, world_size, timeout=None, backend_opts=None):
        """
        Accepts the extra arguments (store, timeout, backend_opts) that the
        PyTorch distributed framework passes in during initialization.
        """
        super().__init__()
        self.store = store
        self.rank = rank
        self.world_size = world_size
        self.timeout = timeout
        self.backend_opts = backend_opts

    def allreduce(self, tensor, opts=None):
        # For demonstration, we “simulate” an allreduce by simply cloning the tensor.
        # In a multi–process environment, you would sum (or reduce) the tensor across ranks.
        print(f"[Rank {self.rank}] Called allreduce on tensor: {tensor}")
        result = tensor.clone()
        return CustomWork(result)

    def broadcast(self, tensor, opts=None):
        # Simulate broadcast by cloning the tensor.
        print(f"[Rank {self.rank}] Called broadcast on tensor: {tensor}")
        result = tensor.clone()
        return CustomWork(result)

    def allgather(self, tensor_list, tensor, opts=None):
        # Simulate allgather by clearing and appending the input tensor.
        print(f"[Rank {self.rank}] Called allgather on tensor: {tensor}")
        tensor_list.clear()
        tensor_list.append(tensor.clone())
        return CustomWork(tensor_list)

###############################################################################
# 3. Register the custom backend with the distributed module.
###############################################################################
# The register_backend API takes a backend name and a ProcessGroup class.
dist.Backend.register_backend("custom", CustomProcessGroup)

###############################################################################
# 4. Define a Triton kernel (if available) and a fallback for CPU testing.
###############################################################################
try:
    import triton
    import triton.language as tl

    @triton.jit
    def triton_kernel(X, Y, N: tl.constexpr):
        # A simple kernel: each thread loads an element from X, multiplies by 2, and writes to Y.
        pid = tl.program_id(0)
        BLOCK_SIZE = 128  # number of elements per block
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x_vals = tl.load(X + offsets, mask=mask)
        tl.store(Y + offsets, x_vals * 2, mask=mask)

    def run_triton_kernel(x):
        # Launch the Triton kernel.
        N = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        triton_kernel[grid](x, y, N, BLOCK_SIZE=128)
        return y

except ImportError:
    print("Triton not available; falling back to CPU multiplication.")

    def run_triton_kernel(x):
        # Fallback “kernel” for CPU: simply multiply by 2.
        return x * 2

###############################################################################
# 5. Define a function that uses our “Triton” kernel and calls a distributed op.
###############################################################################
def compute_and_communicate(x):
    # Use the (compiled) Triton kernel to compute y = x * 2.
    y = run_triton_kernel(x)
    # Now “allreduce” y. (For a single process, our custom backend just returns a clone.)
    work = dist.all_reduce(y)
    y = work.wait()
    return y

###############################################################################
# 6. Main driver: initialize the custom process group, compile the function, and run it.
###############################################################################
def main():
    # For easy CPU testing, we set rank=0 and world_size=1.
    rank = 0
    world_size = 1

    # Initialize the process group using our custom backend.
    # (In a real multi–process job you would pass an appropriate init_method.)
    dist.init_process_group(backend="custom", rank=rank, world_size=world_size)

    # Create an input tensor.
    x = torch.tensor([1.0, 2.0, 3.0])
    print("Input tensor:", x)

    # Use torch.compile to optimize the function.
    compiled_func = torch.compile(compute_and_communicate)

    # Run the compiled function.
    result = compiled_func(x)
    print("Result tensor:", result)

    # Cleanup the process group.
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
