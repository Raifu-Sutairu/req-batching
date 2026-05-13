from dataclasses import dataclass, field
from typing import List

@dataclass
class Step:
    batch_size: int
    batch_age_ms: float
    upstream_p99_ms: float
    request_rate: float
    
    def to_kwargs(self):
        return {
            "batch_size": self.batch_size,
            "batch_age_ms": self.batch_age_ms,
            "upstream_p99_ms": self.upstream_p99_ms,
            "request_rate": self.request_rate,
        }

@dataclass
class Episode:
    batch_key: str
    steps: List[Step] = field(default_factory=list)
    flush_reason: str = ""
    timestamp_unix: int = 0
