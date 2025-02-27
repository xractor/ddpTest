"""
Modern implementation of a dummy backend for PyTorch 2.0+.
This version follows PyTorch's newer ProcessGroup API.
"""
import os
import datetime
import torch
import torch.distributed as dist
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dummy_backend")

class DummyWork(dist.Work):
    """Work object for the dummy backend."""
    
    def __init__(self):
        """Initialize a work object."""
        self._completed = True
        
    def is_completed(self):
        """Return True as work is always completed."""
        return True
        
    def is_success(self):
        """Return True as work is always successful."""
        return True
        
    def wait(self, timeout=None):
        """Wait for completion (always returns True)."""
        return True
        
    def exception(self):
        """Return None as there's never an exception."""
        return None

class DummyProcessGroup(dist.ProcessGroup):
    """Process group implementation for the dummy backend."""
    
    def __init__(self, rank, size):
        """
        Initialize the dummy process group.
        
        Args:
            rank: The rank of this process
            size: The world size
        """
        super().__init__(rank, size)
        self.rank = rank
        self.size = size
        logger.info(f"Created DummyProcessGroup with rank={rank}, size={size}")
        
    def getBackendName(self):
        """Return the backend name."""
        return "DUMMY"
        
    def allreduce(self, tensors, opts=None):
        """
        Dummy implementation of allreduce that sets tensors to zero.
        
        Args:
            tensors: List of tensors to reduce
            opts: Options for the operation
            
        Returns:
            A work object
        """
        logger.info(f"Performing allreduce on {len(tensors)} tensors")
        for tensor in tensors:
            tensor.zero_()
        return DummyWork()
    
    def allgather(self, output_tensors, input_tensors, opts=None):
        """
        Dummy implementation of allgather that sets output tensors to zero.
        
        Args:
            output_tensors: List of lists of output tensors
            input_tensors: List of input tensors
            opts: Options for the operation
            
        Returns:
            A work object
        """
        logger.info(f"Performing allgather on {len(input_tensors)} input tensors")
        for output_list in output_tensors:
            for tensor in output_list:
                tensor.zero_()
        return DummyWork()


# Create a helper function for initializing the process group
def init_dummy_pg(store, rank, world_size, timeout=datetime.timedelta(seconds=300)):
    """
    Helper function to create a dummy process group.
    
    Args:
        store: The store for the process group
        rank: The rank of this process
        world_size: The world size
        timeout: Timeout for operations
        
    Returns:
        A dummy process group
    """
    logger.info(f"Creating dummy process group with rank={rank}, world_size={world_size}")
    return DummyProcessGroup(rank, world_size)


# Register the backend
def register_modern_dummy_backend():
    """Register the dummy backend with PyTorch."""
    logger.info("Registering modern dummy backend...")
    
    # Check which registration method is available
    registration_successful = False
    
    # Try standard method (PyTorch 1.8+)
    if hasattr(dist, "register_backend"):
        try:
            dist.register_backend("dummy", init_dummy_pg)
            logger.info("Successfully registered using dist.register_backend")
            registration_successful = True
        except Exception as e:
            logger.error(f"Error registering with dist.register_backend: {e}")
    
    # Try alternative method for older PyTorch
    if not registration_successful and hasattr(torch.distributed, "_register_backend"):
        try:
            torch.distributed._register_backend("dummy", init_dummy_pg)
            logger.info("Successfully registered using _register_backend")
            registration_successful = True
        except Exception as e:
            logger.error(f"Error registering with _register_backend: {e}")
    
    if not registration_successful:
        logger.error("Could not register backend with any available method")
        available_methods = [m for m in dir(torch.distributed) if "register" in m]
        logger.info(f"Available registration methods: {available_methods}")
    
    # Verify registration
    if "DUMMY" in dist.Backend.__members__:
        logger.info("Verified: DUMMY is in Backend.__members__")
    else:
        logger.warning("DUMMY not found in Backend.__members__")
        logger.info(f"Available backends: {list(dist.Backend.__members__.keys())}")


# Example usage
if __name__ == "__main__":
    # Register backend
    register_modern_dummy_backend()
    
    # Set environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    
    # Initialize process group
    try:
        logger.info("Initializing process group...")
        dist.init_process_group(backend="dummy")
        
        # Test with a tensor
        tensor = torch.ones(5, 5)
        logger.info(f"Tensor before allreduce: {tensor[0][0]}")
        
        dist.all_reduce(tensor)
        logger.info(f"Tensor after allreduce: {tensor[0][0]}")
        
        # Clean up
        dist.destroy_process_group()
        logger.info("Example completed successfully!")
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)
