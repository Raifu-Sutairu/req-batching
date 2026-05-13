import os
import tomllib
from pathlib import Path

class Config:
    def __init__(self, toml_path="../reverse-proxy/config.toml"):
        self.max_batch_size = 128
        self.batch_timeout_ms = 50.0
        self.kafka_brokers = "127.0.0.1:9092"
        self.kafka_telemetry_topic = "batch-telemetry"
        
        # Load from TOML
        toml_file = Path(toml_path)
        if toml_file.exists():
            with open(toml_file, "rb") as f:
                data = tomllib.load(f)
                self.max_batch_size = data.get("max_batch_size", self.max_batch_size)
                self.batch_timeout_ms = float(data.get("batch_timeout_ms", self.batch_timeout_ms))
                self.kafka_brokers = data.get("kafka_brokers", self.kafka_brokers)
                self.kafka_telemetry_topic = data.get("kafka_telemetry_topic", self.kafka_telemetry_topic)

        # Override with env vars matching the Rust proxy's behaviour
        self.max_batch_size = int(os.environ.get("PROXY_MAX_BATCH_SIZE", self.max_batch_size))
        self.batch_timeout_ms = float(os.environ.get("PROXY_BATCH_TIMEOUT_MS", self.batch_timeout_ms))
        self.kafka_brokers = os.environ.get("PROXY_KAFKA_BROKERS", self.kafka_brokers)
        self.kafka_telemetry_topic = os.environ.get("PROXY_KAFKA_TELEMETRY_TOPIC", self.kafka_telemetry_topic)

# Singleton instance
config = Config()
