"""Hardware detection and GPU configuration for Genesis."""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    total_memory: int  # in bytes
    free_memory: int
    compute_capability: tuple[int, int]

    @property
    def total_memory_gb(self) -> float:
        """Return total memory in GB."""
        return self.total_memory / (1024**3)

    @property
    def free_memory_gb(self) -> float:
        """Return free memory in GB."""
        return self.free_memory / (1024**3)

    def __str__(self) -> str:
        return (
            f"GPU {self.index}: {self.name} "
            f"({self.free_memory_gb:.1f}/{self.total_memory_gb:.1f} GB free)"
        )


class HardwareConfig:
    """Hardware configuration and GPU management."""

    def __init__(
        self,
        teacher_device: str = "cuda:0",
        student_device: str = "cuda:1",
        auto_detect: bool = True,
    ):
        self.teacher_device = teacher_device
        self.student_device = student_device
        self._gpus: list[GPUInfo] = []
        self._cuda_available = False
        self._initialized = False

        if auto_detect:
            self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detect available hardware."""
        try:
            import torch

            self._cuda_available = torch.cuda.is_available()

            if self._cuda_available:
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    gpu_info = GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory=total_mem,
                        free_memory=free_mem,
                        compute_capability=(props.major, props.minor),
                    )
                    self._gpus.append(gpu_info)
                    logger.info(f"Detected: {gpu_info}")

                # Validate device assignments
                self._validate_devices()
            else:
                logger.warning("CUDA not available. Using CPU.")
                self.teacher_device = "cpu"
                self.student_device = "cpu"

            self._initialized = True

        except ImportError:
            logger.error("PyTorch not installed. Cannot detect hardware.")
            self._cuda_available = False

    def _validate_devices(self) -> None:
        """Validate that configured devices are available."""
        num_gpus = len(self._gpus)

        def parse_device(device: str) -> Optional[int]:
            if device == "cpu":
                return None
            if device.startswith("cuda:"):
                return int(device.split(":")[1])
            if device == "cuda":
                return 0
            return None

        teacher_idx = parse_device(self.teacher_device)
        student_idx = parse_device(self.student_device)

        if teacher_idx is not None and teacher_idx >= num_gpus:
            logger.warning(
                f"Teacher device {self.teacher_device} not available. "
                f"Only {num_gpus} GPU(s) detected. Using cuda:0."
            )
            self.teacher_device = "cuda:0" if num_gpus > 0 else "cpu"

        if student_idx is not None and student_idx >= num_gpus:
            logger.warning(
                f"Student device {self.student_device} not available. "
                f"Only {num_gpus} GPU(s) detected."
            )
            # Fall back to same device as teacher or CPU
            if num_gpus >= 2:
                self.student_device = "cuda:1"
            elif num_gpus == 1:
                self.student_device = "cuda:0"
            else:
                self.student_device = "cpu"

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def num_gpus(self) -> int:
        """Return number of available GPUs."""
        return len(self._gpus)

    @property
    def gpus(self) -> list[GPUInfo]:
        """Return list of GPU information."""
        return self._gpus

    @property
    def dual_gpu_available(self) -> bool:
        """Check if dual GPU setup is available."""
        return self.num_gpus >= 2

    def get_gpu(self, index: int) -> Optional[GPUInfo]:
        """Get GPU info by index."""
        if 0 <= index < len(self._gpus):
            return self._gpus[index]
        return None

    def get_optimal_batch_size(self, model_memory_gb: float, device: str) -> int:
        """Estimate optimal batch size based on available memory."""
        if device == "cpu":
            return 1

        device_idx = int(device.split(":")[1]) if ":" in device else 0
        gpu = self.get_gpu(device_idx)

        if gpu is None:
            return 1

        # Reserve some memory for PyTorch overhead
        available_memory = gpu.free_memory_gb * 0.8
        memory_per_sample = model_memory_gb * 0.1  # Rough estimate

        batch_size = max(1, int(available_memory / memory_per_sample))
        return min(batch_size, 32)  # Cap at 32

    def memory_summary(self) -> str:
        """Return a summary of GPU memory usage."""
        lines = ["GPU Memory Summary:"]
        for gpu in self._gpus:
            lines.append(f"  {gpu}")
        return "\n".join(lines)

    def optimize_device_assignment(self, teacher_memory_gb: float, student_memory_gb: float) -> None:
        """Optimize device assignment based on model sizes and GPU memory."""
        if not self.dual_gpu_available:
            logger.info("Single GPU or CPU mode. Both models on same device.")
            return

        gpu0 = self._gpus[0]
        gpu1 = self._gpus[1]

        # Assign larger model to GPU with more memory
        if teacher_memory_gb > student_memory_gb:
            if gpu0.free_memory_gb >= gpu1.free_memory_gb:
                self.teacher_device = "cuda:0"
                self.student_device = "cuda:1"
            else:
                self.teacher_device = "cuda:1"
                self.student_device = "cuda:0"
        else:
            if gpu0.free_memory_gb >= gpu1.free_memory_gb:
                self.teacher_device = "cuda:1"
                self.student_device = "cuda:0"
            else:
                self.teacher_device = "cuda:0"
                self.student_device = "cuda:1"

        logger.info(f"Optimized device assignment: Teacher={self.teacher_device}, Student={self.student_device}")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "teacher_device": self.teacher_device,
            "student_device": self.student_device,
            "cuda_available": self._cuda_available,
            "num_gpus": self.num_gpus,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_gb": gpu.total_memory_gb,
                    "free_memory_gb": gpu.free_memory_gb,
                    "compute_capability": gpu.compute_capability,
                }
                for gpu in self._gpus
            ],
        }


def get_device(device_str: str) -> "torch.device":
    """Get torch device from string."""
    import torch

    return torch.device(device_str)


def empty_cache(device: Optional[str] = None) -> None:
    """Empty CUDA cache for specified device or all devices."""
    import torch

    if torch.cuda.is_available():
        if device is not None and device.startswith("cuda"):
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()


def synchronize(device: Optional[str] = None) -> None:
    """Synchronize CUDA operations."""
    import torch

    if torch.cuda.is_available():
        if device is not None and device.startswith("cuda"):
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            torch.cuda.synchronize(device_idx)
        else:
            torch.cuda.synchronize()
