import torch
import torch.distributed as dist
from .dummy_backend import create_dummy_process_group

# Register the dummy backend with PyTorch's distributed module
try:
    # Make sure the dummy backend is registered only once
    if "dummy" not in dist.Backend._backend_registry:
        dist.Backend.register_backend("dummy", create_dummy_process_group)
        print("Successfully registered 'dummy' backend with PyTorch distributed")
    
    __all__ = ["create_dummy_process_group"]
except Exception as e:
    print(f"Failed to register dummy backend: {e}")
