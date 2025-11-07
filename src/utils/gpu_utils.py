import torch
from typing import Dict, Optional


def is_gpu_available() -> bool:
    return torch.cuda.is_available()


def get_gpu_info() -> Optional[Dict]:
    if not torch.cuda.is_available():
        return None

    try:
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()

        devices = []
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = props.total_memory

            devices.append(
                {
                    "id": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_total_gb": round(memory_total / (1024**3), 2),
                    "memory_allocated_gb": round(memory_allocated / (1024**3), 2),
                    "memory_reserved_gb": round(memory_reserved / (1024**3), 2),
                    "memory_free_gb": round(
                        (memory_total - memory_reserved) / (1024**3), 2
                    ),
                    "utilization_percent": (
                        round((memory_allocated / memory_total) * 100, 1)
                        if memory_total > 0
                        else 0
                    ),
                }
            )

        return {
            "available": True,
            "device_count": gpu_count,
            "current_device": current_device,
            "cuda_version": torch.version.cuda,
            "devices": devices,
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


def get_device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_index() -> int:
    return 0 if torch.cuda.is_available() else -1
