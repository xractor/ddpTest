import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Dict, Any, Optional, Tuple
import os
import datetime


class _DummyWork(dist.Work):
    """
    Dummy implementation of torch.distributed.Work
    """
    def __init__(self, tensors=None):
        super().__init__()
        self.tensors = tensors
        self._complete = True
        
    def is_completed(self) -> bool:
        return self._complete
        
    def is_success(self) -> bool:
        return True
        
    def exception(self) -> Optional[Exception]:
        return None
    
    def wait(self, timeout: Optional[datetime.timedelta] = None) -> bool:
        return True
    
    def _wait_impl(self, timeout: Optional[datetime.timedelta] = None) -> bool:
        return True
    
    def _get_native_tensors(self):
        return self.tensors


class DummyProcessGroup(ProcessGroup):
    """
    A dummy implementation of ProcessGroup for PyTorch distributed operations.
    """
    def __init__(self, rank: int, size: int, timeout: datetime.timedelta):
        """
        Initialize the DummyProcessGroup.
        
        Args:
            rank (int): The rank of this process
            size (int): The world size (total number of processes)
            timeout (datetime.timedelta): Operation timeout
        """
        super().__init__(rank, size)
        self.rank = rank
        self.size = size
        self.timeout = timeout
        
    def allgather(self, output_tensors_lists, input_tensors, opts=None) -> dist.Work:
        """
        Dummy implementation of allgather that sets all output tensors to zero.
        
        Args:
            output_tensors_lists (List[List[torch.Tensor]]): Output tensors
            input_tensors (List[torch.Tensor]): Input tensors
            opts: Optional options
            
        Returns:
            Work: A work handle
        """
        for output_tensor_list in output_tensors_lists:
            for output_tensor in output_tensor_list:
                output_tensor.zero_()
        
        return _DummyWork(output_tensors_lists)
    
    def _allgather_base(self, output_tensor, input_tensor, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def allreduce(self, tensors, opts=None) -> dist.Work:
        """
        Dummy implementation of allreduce that sets all tensors to zero.
        
        Args:
            tensors (List[torch.Tensor]): The tensors to reduce
            opts: Optional options
            
        Returns:
            Work: A work handle
        """
        for tensor in tensors:
            tensor.zero_()
        
        return _DummyWork(tensors)
    
    def allreduce_coalesced(self, tensors, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def alltoall(self, output_tensors, input_tensors, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def alltoall_base(self, output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def barrier(self, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def broadcast(self, tensors, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def gather(self, output_tensors, input_tensors, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def reduce(self, tensors, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def reduce_scatter(self, output_tensors, input_tensors, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def scatter(self, output_tensors, input_tensors, opts=None) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def send(self, tensors, dst_rank, tag=0) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def recv(self, tensors, src_rank, tag=0) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")
    
    def recv_anysource(self, tensors, tag=0) -> dist.Work:
        """
        Not supported.
        """
        raise RuntimeError("not supported")


def create_dummy_process_group(
    store, 
    rank: int, 
    size: int, 
    timeout: datetime.timedelta = datetime.timedelta(seconds=30)
) -> DummyProcessGroup:
    """
    Factory function to create a DummyProcessGroup instance.
    
    Args:
        store: The store used for rendezvous
        rank (int): The rank of this process
        size (int): The world size (total number of processes)
        timeout (datetime.timedelta): Operation timeout
        
    Returns:
        DummyProcessGroup: An instance of the dummy process group
    """
    return DummyProcessGroup(rank, size, timeout)
