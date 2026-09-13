"""
Environment diagnostics for FoveaLiDAR.

Detects hardware capabilities, Python environment, and installed packages
to enable compute-aware operation and graceful degradation.
"""

from __future__ import annotations

import platform
import sys
import subprocess
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

try:
    from PySide6 import QtCore
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class CPUInfo:
    """CPU information."""
    model: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: float
    architecture: str


@dataclass
class MemoryInfo:
    """Memory information."""
    total_gb: float
    available_gb: float
    percent_used: float


@dataclass
class GPUInfo:
    """GPU information."""
    name: str
    vram_total_gb: float
    vram_free_gb: float
    driver_version: str
    cuda_version: Optional[str] = None
    compute_capability: Optional[str] = None
    is_integrated: bool = False


@dataclass
class EnvironmentInfo:
    """Complete environment information."""
    os: str
    os_version: str
    python_version: str
    platform: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: List[GPUInfo] = field(default_factory=list)
    packages: Dict[str, str] = field(default_factory=dict)
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    tensorrt_available: bool = False
    warnings: List[str] = field(default_factory=list)


def get_cpu_info() -> CPUInfo:
    """Get CPU information."""
    try:
        if platform.system() == "Windows":
            import wmi
            c = wmi.WMI()
            cpu_model = c.Win32_Processor()[0].Name
        else:
            cpu_model = platform.processor()
            if not cpu_model:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            cpu_model = line.split(':')[1].strip()
                            break
    except Exception:
        cpu_model = platform.processor() or "Unknown"
    
    return CPUInfo(
        model=cpu_model,
        cores_physical=psutil.cpu_count(logical=False) or 0,
        cores_logical=psutil.cpu_count(logical=True) or 0,
        frequency_mhz=psutil.cpu_freq().current if psutil.cpu_freq() else 0.0,
        architecture=platform.machine()
    )


def get_memory_info() -> MemoryInfo:
    """Get memory information."""
    mem = psutil.virtual_memory()
    return MemoryInfo(
        total_gb=mem.total / (1024**3),
        available_gb=mem.available / (1024**3),
        percent_used=mem.percent
    )


def get_gpu_info() -> List[GPUInfo]:
    """Get GPU information using nvidia-smi."""
    gpus = []
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,compute_cap', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    name = parts[0]
                    vram_total = float(parts[1]) / 1024
                    vram_free = float(parts[2]) / 1024
                    driver_version = parts[3]
                    compute_cap = parts[4] if len(parts) > 4 else "Unknown"
                    
                    gpus.append(GPUInfo(
                        name=name,
                        vram_total_gb=vram_total,
                        vram_free_gb=vram_free,
                        driver_version=driver_version,
                        compute_capability=compute_cap
                    ))
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    
    if TORCH_AVAILABLE and torch.cuda.is_available() and not gpus:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                name=props.name,
                vram_total_gb=props.total_memory / (1024**3),
                vram_free_gb=(props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3),
                driver_version="Unknown",
                cuda_version=torch.version.cuda,
                compute_capability=f"{props.major}.{props.minor}"
            ))
    
    return gpus


def get_cuda_info() -> tuple[bool, Optional[str], Optional[str]]:
    """Get CUDA availability and versions."""
    cuda_available = False
    cuda_version = None
    cudnn_version = None
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        cuda_available = True
        cuda_version = torch.version.cuda
        cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
    
    return cuda_available, cuda_version, cudnn_version


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    
    package_names = [
        'numpy', 'torch', 'torchvision', 'open3d', 'PySide6',
        'yaml', 'psutil', 'pytest'
    ]
    
    for name in package_names:
        try:
            if name == 'yaml':
                import yaml
                packages[name] = yaml.__version__
            elif name == 'torch':
                if TORCH_AVAILABLE:
                    packages[name] = torch.__version__
            elif name == 'open3d':
                if OPEN3D_AVAILABLE:
                    packages[name] = o3d.__version__
            elif name == 'PySide6':
                if PYSIDE6_AVAILABLE:
                    from PySide6 import __version__ as pyside_version
                    packages[name] = pyside_version
            else:
                module = __import__(name)
                packages[name] = getattr(module, '__version__', 'Unknown')
        except (ImportError, AttributeError):
            packages[name] = "Not installed"
    
    return packages


def check_tensorrt() -> bool:
    """Check if TensorRT is available."""
    try:
        import tensorrt
        return True
    except ImportError:
        return False


def run_diagnostics() -> EnvironmentInfo:
    """Run complete environment diagnostics."""
    warnings = []
    
    # Gather all diagnostics
    cpu = get_cpu_info()
    memory = get_memory_info()
    gpus = get_gpu_info()
    cuda_available, cuda_version, cudnn_version = get_cuda_info()
    packages = get_package_versions()
    tensorrt_available = check_tensorrt()
    
    # Generate warnings
    if memory.percent_used > 80:
        warnings.append(f"WARNING: Memory usage is high ({memory.percent_used:.1f}%)")
    
    if not gpus and TORCH_AVAILABLE:
        warnings.append("WARNING: No GPU detected; CPU mode will be slower")
    
    if not OPEN3D_AVAILABLE:
        warnings.append("WARNING: Open3D not available; 3D visualization disabled")
    
    if not PYSIDE6_AVAILABLE:
        warnings.append("WARNING: PySide6 not available; GUI disabled")
    
    if not YAML_AVAILABLE:
        warnings.append("WARNING: PyYAML not available; config loading disabled")
    
    return EnvironmentInfo(
        os=platform.system(),
        os_version=platform.release(),
        python_version=sys.version,
        platform=platform.platform(),
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        packages=packages,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        tensorrt_available=tensorrt_available,
        warnings=warnings
    )


def print_diagnostics(info: EnvironmentInfo) -> None:
    """Pretty-print diagnostics to console."""
    print("\n" + "=" * 80)
    print("FoveaLiDAR Environment Diagnostics")
    print("=" * 80)
    
    print(f"\n[System]")
    print(f"  OS: {info.os} {info.os_version}")
    print(f"  Platform: {info.platform}")
    print(f"  Python: {info.python_version.split()[0]}")
    
    print(f"\n[CPU]")
    print(f"  Model: {info.cpu.model}")
    print(f"  Cores: {info.cpu.cores_physical} physical, {info.cpu.cores_logical} logical")
    print(f"  Frequency: {info.cpu.frequency_mhz:.0f} MHz")
    print(f"  Architecture: {info.cpu.architecture}")
    
    print(f"\n[Memory]")
    print(f"  Total: {info.memory.total_gb:.2f} GB")
    print(f"  Available: {info.memory.available_gb:.2f} GB")
    print(f"  Used: {info.memory.percent_used:.1f}%")
    
    if info.gpus:
        print(f"\n[GPU] ({len(info.gpus)} device(s))")
        for i, gpu in enumerate(info.gpus):
            print(f"  GPU {i}: {gpu.name}")
            print(f"    VRAM: {gpu.vram_total_gb:.2f} GB total, {gpu.vram_free_gb:.2f} GB free")
            print(f"    Driver: {gpu.driver_version}")
            if gpu.compute_capability:
                print(f"    Compute Capability: {gpu.compute_capability}")
    else:
        print(f"\n[GPU]")
        print(f"  No GPUs detected")
    
    if info.cuda_available:
        print(f"\n[CUDA]")
        print(f"  Available: Yes")
        if info.cuda_version:
            print(f"  Version: {info.cuda_version}")
        if info.cudnn_version:
            print(f"  cuDNN: {info.cudnn_version}")
        if info.tensorrt_available:
            print(f"  TensorRT: Available")
    else:
        print(f"\n[CUDA]")
        print(f"  Available: No")
    
    print(f"\n[Packages]")
    for name, version in sorted(info.packages.items()):
        print(f"  {name}: {version}")
    
    if info.warnings:
        print(f"\n[Warnings] ({len(info.warnings)})")
        for warning in info.warnings:
            print(f"  - {warning}")
    
    print("\n" + "=" * 80 + "\n")


def get_compute_recommendation(info: EnvironmentInfo) -> str:
    """Get compute mode recommendation based on environment."""
    if not info.gpus:
        return "cpu"
    
    # Check VRAM
    total_vram = sum(gpu.vram_total_gb for gpu in info.gpus)
    if total_vram >= 8:
        return "gpu"
    elif total_vram >= 4:
        return "balanced"
    else:
        return "cpu"


def export_diagnostics_json(info: EnvironmentInfo, filepath: str) -> None:
    """Export diagnostics to JSON file."""
    data = {
        "os": info.os,
        "os_version": info.os_version,
        "python_version": info.python_version,
        "platform": info.platform,
        "cpu": {
            "model": info.cpu.model,
            "cores_physical": info.cpu.cores_physical,
            "cores_logical": info.cpu.cores_logical,
            "frequency_mhz": info.cpu.frequency_mhz,
            "architecture": info.cpu.architecture,
        },
        "memory": {
            "total_gb": info.memory.total_gb,
            "available_gb": info.memory.available_gb,
            "percent_used": info.memory.percent_used,
        },
        "gpus": [
            {
                "name": gpu.name,
                "vram_total_gb": gpu.vram_total_gb,
                "vram_free_gb": gpu.vram_free_gb,
                "driver_version": gpu.driver_version,
                "compute_capability": gpu.compute_capability,
            }
            for gpu in info.gpus
        ],
        "packages": info.packages,
        "cuda_available": info.cuda_available,
        "cuda_version": info.cuda_version,
        "cudnn_version": info.cudnn_version,
        "tensorrt_available": info.tensorrt_available,
        "warnings": info.warnings,
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    info = run_diagnostics()
    print_diagnostics(info)
    print(f"Recommended compute mode: {get_compute_recommendation(info)}")
    export_diagnostics_json(info, "diagnostics.json")