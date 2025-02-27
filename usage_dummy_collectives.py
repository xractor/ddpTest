import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any, Union
import datetime

class WorkDummy:
    """
    A dummy work object similar to those returned by PyTorch distributed operations.
    
    This represents an asynchronous operation that is always completed.
    """
    def __init__(self, op_type, future=None):
        self.op_type = op_type
        self._future = future if future is not None else torch.futures.Future()
        if future is None:
            self._future.set_result(None)  # Immediately complete
    
    def is_completed(self):
        """Check if the operation is completed (always returns True)"""
        return True
    
    def is_success(self):
        """Check if the operation was successful (always returns True)"""
        return True
    
    def wait(self, timeout=None):
        """Wait for the operation to complete (always returns immediately)"""
        return True
    
    def get_future(self):
        """Get the future for the operation"""
        return self._future

class BackendDummy:
    """
    A dummy implementation of collective operations similar to PyTorch distributed.
    
    This class provides methods that match the interface of PyTorch's distributed
    backend, but with dummy implementations that don't perform actual communication.
    """
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size
    
    def allreduce(self, tensors, opts=None):
        """Zero out all tensors as a dummy allreduce operation"""
        for tensor in tensors:
            tensor.zero_()
        
        future = torch.futures.Future()
        future.set_result(tensors)
        return WorkDummy("ALLREDUCE", future)
    
    def allgather(self, output_tensors, input_tensors, opts=None):
        """Zero out all output tensors as a dummy allgather operation"""
        for output_tensor_vec in output_tensors:
            for tensor in output_tensor_vec:
                tensor.zero_()
        
        future = torch.futures.Future()
        future.set_result(output_tensors)
        return WorkDummy("ALLGATHER", future)
    
    def _allgather_base(self, output_buffer, input_buffer, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def allreduce_coalesced(self, tensors, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def broadcast(self, tensors, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def reduce(self, tensors, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def alltoall_base(self, output_tensor, input_tensor, output_split_sizes=None, input_split_sizes=None, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def alltoall(self, output_tensors, input_tensors, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def barrier(self, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def gather(self, output_tensors, input_tensors, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def scatter(self, output_tensors, input_tensors, opts=None):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def send(self, tensors, dst_rank, tag):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def recv(self, tensors, src_rank, tag):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")
    
    def recvAnysource(self, tensors, tag):
        """Not implemented - raise error like the C++ version"""
        raise RuntimeError("not supported")

def create_backend_dummy(store, rank, size, timeout=datetime.timedelta(seconds=30)):
    """
    Factory function to create a dummy backend.
    
    Args:
        store: Store for distributed communication (unused)
        rank (int): Rank of the current process
        size (int): Total number of processes
        timeout: Timeout for operations (unused)
        
    Returns:
        BackendDummy: An instance of the dummy backend
    """
    return BackendDummy(rank, size)

# Helper function to initialize the dummy backend directly
def init_dummy(rank=0, size=1):
    """
    Initialize a dummy backend directly.
    
    Args:
        rank (int): Rank of the current process
        size (int): Total number of processes
        
    Returns:
        BackendDummy: An instance of the dummy backend
    """
    return BackendDummy(rank, size)
