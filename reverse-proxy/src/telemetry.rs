use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
pub struct BatchFlushEvent {
    pub batch_key: String,
    pub batch_size: usize,
    pub batch_age_ms: f64,
    pub upstream_ms: f64,
    pub flush_reason: String,
    pub timestamp_unix: u64,
}

pub struct TelemetryPublisher {
    producer: FutureProducer,
    topic: String,
}

impl TelemetryPublisher {
    pub fn new(brokers: &str, topic: &str) -> Result<Self, rdkafka::error::KafkaError> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .set("message.timeout.ms", "5000")
            .create()?;

        Ok(Self {
            producer,
            topic: topic.to_string(),
        })
    }

    pub async fn publish_flush(&self, event: BatchFlushEvent) {
        match serde_json::to_string(&event) {
            Ok(payload) => {
                let record = FutureRecord::to(&self.topic)
                    .payload(&payload)
                    .key(&event.batch_key);

                match self.producer.send(record, rdkafka::util::Timeout::Never).await {
                    Ok(_) => tracing::debug!("Telemetry event published successfully"),
                    Err((e, _)) => tracing::warn!("Failed to publish telemetry event: {}", e),
                }
            }
            Err(e) => tracing::warn!("Failed to serialize telemetry event: {}", e),
        }
    }
}
