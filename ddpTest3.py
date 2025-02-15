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
        # For demonstration, simulate an allreduce by simply cloning the tensor.
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
# Register our custom process group under the name "custom".
dist.Backend.register_backend("custom", CustomProcessGroup)

###############################################################################
# 4. Define a Triton kernel (if available and if a GPU is present) and a fallback.
###############################################################################
# Check if CUDA is available. Even if Triton is installed, launching its kernel
# without an active GPU driver will trigger the error "0 active drivers([])..."
if torch.cuda.is_available():
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def triton_kernel(X, Y, N: tl.constexpr):
            # Each thread loads an element from X, multiplies by 2, and writes to Y.
            pid = tl.program_id(0)
            BLOCK_SIZE = 128  # number of elements per block
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            x_vals = tl.load(X + offsets, mask=mask)
            tl.store(Y + offsets, x_vals * 2, mask=mask)

        def run_triton_kernel(x):
            N = x.numel()
            # Ensure that the output tensor is allocated on the same device as x.
            y = torch.empty_like(x, device=x.device)
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
            triton_kernel[grid](x, y, N, BLOCK_SIZE=128)
            return y

    except Exception as e:
        print(f"Triton encountered an error: {e}; falling back to CPU multiplication.")
        def run_triton_kernel(x):
            return x * 2
else:
    print("No CUDA device found; using CPU fallback for Triton kernel.")
    def run_triton_kernel(x):
        # Fallback “kernel” for CPU: simply multiply by 2.
        return x * 2

###############################################################################
# 5. Define a function that uses our “Triton” kernel and calls a distributed op.
###############################################################################
def compute_and_communicate(x):
    # Use the Triton kernel (or fallback) to compute y = x * 2.
    y = run_triton_kernel(x)
    # Now call allreduce (which just clones the tensor in our custom backend).
    work = dist.all_reduce(y)
    y = work.wait()
    return y

###############################################################################
# 6. Main driver: initialize the custom process group, compile the function, and run it.
###############################################################################
def main():
    # For testing on a single process, set rank=0 and world_size=1.
    rank = 0
    world_size = 1

    # Initialize the process group using our custom backend.
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
