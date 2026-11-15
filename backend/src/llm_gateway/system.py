"""Host information: execution environment and GPU memory."""

import asyncio
import platform
from typing import Literal

from pydantic import BaseModel

Environment = Literal["wsl", "linux", "other"]

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


class GpuStatus(BaseModel):
    available: bool
    name: str | None = None
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    utilization_percent: int | None = None
    error: str | None = None


def memory_mb() -> tuple[int | None, int | None]:
    """(available, total) system memory in MB, or (None, None) where it
    cannot be read. "Available" is what the kernel says can be used without
    swapping; inside WSL it refers to the WSL virtual machine."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            values = {line.split(":")[0]: int(line.split()[1]) for line in meminfo}
        return values["MemAvailable"] // 1024, values["MemTotal"] // 1024
    except (OSError, KeyError, ValueError, IndexError):
        return None, None


def memory_available_mb() -> int | None:
    return memory_mb()[0]


def detect_environment() -> Environment:
    if platform.system() != "Linux":
        return "other"
    return "wsl" if "microsoft" in platform.release().lower() else "linux"


def parse_nvidia_smi(output: str) -> GpuStatus:
    """Parse the first GPU line of NVIDIA_SMI_QUERY output."""
    lines = output.strip().splitlines()
    fields = [f.strip() for f in lines[0].split(",")] if lines else []
    try:
        name, used, total, utilization = fields
        return GpuStatus(
            available=True,
            name=name,
            memory_used_mib=int(used),
            memory_total_mib=int(total),
            utilization_percent=int(utilization),
        )
    except ValueError:
        return GpuStatus(available=False, error="Unrecognised nvidia-smi output")


async def gpu_status(timeout_seconds: float = 3.0) -> GpuStatus:
    """Query nvidia-smi. Never raises: any failure becomes `available=False`."""
    try:
        process = await asyncio.create_subprocess_exec(
            *NVIDIA_SMI_QUERY,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return GpuStatus(available=False, error="nvidia-smi not found")
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass  # it exited just as the time ran out
        return GpuStatus(available=False, error="nvidia-smi timed out")
    if process.returncode != 0:
        return GpuStatus(
            available=False, error=stderr.decode(errors="replace").strip()[:200] or "failed"
        )
    return parse_nvidia_smi(stdout.decode(errors="replace"))
