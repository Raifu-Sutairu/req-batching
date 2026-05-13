import json
import logging
from confluent_kafka import Consumer, KafkaError
from collections import deque
from typing import List
import sys
from pathlib import Path

# Add parent directory to path so we can import from rl.*
sys.path.append(str(Path(__file__).parent.parent))
from data.episode import Episode, Step
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelemetryConsumer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
        conf = {
            'bootstrap.servers': config.kafka_brokers,
            'group.id': 'rl-training-group',
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(conf)
        self.consumer.subscribe([config.kafka_telemetry_topic])
        
        self.p99_tracker = {}

    def _reconstruct_episode(self, payload: dict) -> Episode:
        batch_size = payload.get("batch_size", 1)
        batch_age_ms = payload.get("batch_age_ms", 0.0)
        upstream_ms = payload.get("upstream_ms", 5.0)
        batch_key = payload.get("batch_key", "unknown")
        
        # Exponential moving average to simulate p99
        if batch_key not in self.p99_tracker:
            self.p99_tracker[batch_key] = upstream_ms
        else:
            self.p99_tracker[batch_key] = 0.9 * self.p99_tracker[batch_key] + 0.1 * upstream_ms
            
        p99 = self.p99_tracker[batch_key]
        
        # Interpolate the steps of the episode
        steps = []
        for i in range(1, batch_size + 1):
            fraction = i / batch_size
            steps.append(Step(
                batch_size=i,
                batch_age_ms=batch_age_ms * fraction,
                upstream_p99_ms=p99,
                request_rate=50.0  # Synthetic moderate rate
            ))
            
        return Episode(
            batch_key=batch_key,
            steps=steps,
            flush_reason=payload.get("flush_reason", "unknown"),
            timestamp_unix=payload.get("timestamp_unix", 0)
        )

    def consume_batch(self, max_messages=1000, timeout_s=1.0) -> int:
        added = 0
        msgs = self.consumer.consume(num_messages=max_messages, timeout=timeout_s)
        
        for msg in msgs:
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                    continue
                    
            try:
                payload = json.loads(msg.value().decode('utf-8'))
                ep = self._reconstruct_episode(payload)
                self.buffer.append(ep)
                added += 1
            except Exception as e:
                logger.error(f"Failed to parse msg: {e}")
                
        return added
        
    def close(self):
        self.consumer.close()
