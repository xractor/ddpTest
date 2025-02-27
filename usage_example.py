import torch
from dummy_collectives import init_dummy

def main():
    """
    Example of how to use the dummy backend.
    """
    # Initialize the dummy backend
    dummy_backend = init_dummy(rank=0, size=2)
    
    # Create some tensors for testing
    tensor = torch.ones(5, 5)
    print(f"Original tensor: {tensor}")
    
    # Use allreduce (will zero out the tensor)
    work = dummy_backend.allreduce([tensor])
    work.wait()  # Wait for completion (though it's immediate in this dummy implementation)
    
    print(f"After allreduce: {tensor}")
    
    # Try another operation that's not supported
    try:
        output_tensors = []
        input_tensors = []
        dummy_backend.broadcast(tensors=[tensor])
    except RuntimeError as e:
        print(f"Expected error for unsupported operation: {e}")
    
    # Test allgather
    output_tensors = [[torch.ones(5, 5) for _ in range(2)]]
    input_tensors = [torch.ones(5, 5)]
    
    print(f"Before allgather: {output_tensors[0][0]}")
    
    work = dummy_backend.allgather(output_tensors, input_tensors)
    work.wait()
    
    print(f"After allgather: {output_tensors[0][0]}")

if __name__ == "__main__":
    main()
