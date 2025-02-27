import torch
import torch.distributed as dist
import os

# Import your package - this will automatically register the backend
import dummy_collectives

def run_example():
    """
    Run a simple example using the dummy backend.
    
    This example initializes the process group with the dummy backend
    and performs a simple allreduce operation.
    """
    # Set environment variables for the distributed process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # Initialize process group with dummy backend
    dist.init_process_group(
        backend="dummy",
        world_size=2,
        rank=0
    )
    
    # Create a tensor of ones
    tensor = torch.ones(10, 10)
    print(f"Before allreduce: {tensor[0][0]}")
    
    # Perform allreduce - this should zero the tensor with our dummy backend
    dist.all_reduce(tensor)
    
    # The tensor should now be zeros
    print(f"After allreduce: {tensor[0][0]}")
    
    # Clean up
    dist.destroy_process_group()
    
    print("Example completed successfully!")

if __name__ == "__main__":
    run_example()
