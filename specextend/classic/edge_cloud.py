import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class NetworkConfig:
    rtt_ms: float = 20.0
    uplink_mbps: float = 1000.0
    downlink_mbps: float = 1000.0
    enabled: bool = True


@dataclass
class AsyncConfig:
    enabled: bool = True
    draft_probe_tokens: int = 1
    poll_interval_ms: float = 1.0


@dataclass
class EdgeCloudConfig:
    edge_device: str = "cuda:0"
    cloud_device: str = "cuda:1"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    async_pipeline: AsyncConfig = field(default_factory=AsyncConfig)
    metrics_output: Optional[str] = None

    @classmethod
    def from_file(cls, path: str) -> "EdgeCloudConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        network = NetworkConfig(**data.get("network", {}))
        async_pipeline = AsyncConfig(**data.get("async_pipeline", {}))
        return cls(
            edge_device=data.get("edge_device", "cuda:0"),
            cloud_device=data.get("cloud_device", "cuda:1"),
            network=network,
            async_pipeline=async_pipeline,
            metrics_output=data.get("metrics_output"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgeCloudMetrics:
    def __init__(self):
        self.counters: Dict[str, float] = {}
        self.events = []

    def add(self, key: str, value: float):
        self.counters[key] = self.counters.get(key, 0.0) + float(value)

    def incr(self, key: str, value: int = 1):
        self.counters[key] = self.counters.get(key, 0.0) + int(value)

    def event(self, name: str, **fields):
        item = {"name": name, "t": time.perf_counter()}
        item.update(fields)
        self.events.append(item)

    def summary(self) -> Dict[str, Any]:
        return {
            "counters": self.counters,
            "events": self.events,
        }


def tensor_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_nbytes(v) for v in value)
    return 0


def synchronize_if_cuda(device: Optional[str] = None):
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
            return
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)


class NetworkSimulator:
    def __init__(self, config: NetworkConfig, metrics: Optional[EdgeCloudMetrics] = None):
        self.config = config
        self.metrics = metrics

    def transfer(self, direction: str, payload: Any) -> float:
        if not self.config.enabled:
            return 0.0
        bytes_count = tensor_nbytes(payload)
        if direction == "uplink":
            bandwidth_mbps = self.config.uplink_mbps
        elif direction == "downlink":
            bandwidth_mbps = self.config.downlink_mbps
        else:
            raise ValueError(f"Unknown network direction: {direction}")
        one_way = max(self.config.rtt_ms, 0.0) / 2000.0
        bandwidth = max(bandwidth_mbps, 1e-9) * 1_000_000.0
        serialization = bytes_count * 8.0 / bandwidth
        delay = one_way + serialization
        t0 = time.perf_counter()
        time.sleep(delay)
        elapsed = time.perf_counter() - t0
        if self.metrics is not None:
            self.metrics.add(f"network_{direction}_seconds", elapsed)
            self.metrics.add(f"network_{direction}_bytes", bytes_count)
            self.metrics.incr(f"network_{direction}_transfers")
        return elapsed


def write_metrics(path: str, payload: Dict[str, Any]):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
