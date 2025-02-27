import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime

# Import the dummy backend module
import dummy_collectives


def setup(rank, world_size):
    """
    Setup the distributed environment with the dummy backend.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Create a file store for processes to communicate
    store = dist.FileStore("/tmp/dummy_test_file_store", world_size)
    
    # Initialize the process group with the dummy backend
    process_group = dist.init_process_group(
        "dummy",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30)
    )
    
    print(f"Rank {rank}: Initialized process group with dummy backend")
    return process_group


def cleanup():
    """
    Cleanup the distributed environment.
    """
    dist.destroy_process_group()


def run_demo(rank, world_size):
    """
    Run a demonstration of the dummy backend.
    
    Args:
        rank (int): The rank of this process.
        world_size (int): The total number of processes.
    """
    # Setup the distributed environment
    process_group = setup(rank, world_size)
    
    # Create a tensor to reduce
    tensor = torch.ones(5, device="cpu") * (rank + 1)
    original_tensor = tensor.clone()
    print(f"Rank {rank}: Original tensor: {original_tensor}")
    
    # Perform an allreduce operation
    work = process_group.allreduce([tensor])
    work.wait()
    print(f"Rank {rank}: After allreduce: {tensor}")
    
    # Create tensors for allgather
    tensor = torch.ones(2, device="cpu") * (rank + 1)
    original_tensor = tensor.clone()
    output_tensors = [torch.zeros(2, device="cpu") for _ in range(world_size)]
    print(f"Rank {rank}: Original tensor for allgather: {original_tensor}")
    
    # Perform an allgather operation
    work = process_group.allgather([output_tensors], [tensor])
    work.wait()
    print(f"Rank {rank}: After allgather: {output_tensors}")
    
    # Test the global API as well
    tensor = torch.ones(3, device="cpu") * (rank + 1)
    original_tensor = tensor.clone()
    print(f"Rank {rank}: Original tensor for global API: {original_tensor}")
    dist.all_reduce(tensor, group=process_group)
    print(f"Rank {rank}: After global API all_reduce: {tensor}")
    
    # Try a collective operation that's not implemented
    try:
        tensor = torch.ones(2, device="cpu")
        dist.broadcast(tensor, src=0, group=process_group)
    except RuntimeError as e:
        print(f"Rank {rank}: Expected error for unimplemented operation: {e}")
    
    # Cleanup
    cleanup()


if __name__ == "__main__":
    # Set the world size
    world_size = 2
    
    # Spawn processes
    mp.spawn(run_demo, args=(world_size,), nprocs=world_size, join=True)
