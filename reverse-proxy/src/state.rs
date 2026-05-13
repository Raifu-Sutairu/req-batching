use crate::batch;
use std::sync::Arc;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use http_body_util::Full;
use bytes::Bytes;
use tokio::sync::Mutex as AsyncMutex;
use crate::proto::rl_agent::rl_agent_client::RlAgentClient;
use tonic::transport::Channel;

pub struct LatencyTracker {
    pub window: std::collections::VecDeque<f64>,
    pub capacity: usize,
}

impl LatencyTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn record(&mut self, ms: f64) {
        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(ms);
    }

    pub fn p99(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.window.iter().copied().collect();
        //sort floats safely
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (sorted.len() as f64 * 0.99).ceil() as usize;
        let idx = if idx == 0 { 0 } else { idx - 1 };
        sorted[idx]
    }
}

pub struct RateCounter {
    pub count: usize,
    pub last_reset: std::time::Instant,
}

impl RateCounter {
    pub fn new() -> Self {
        Self {
            count: 0,
            last_reset: std::time::Instant::now(),
        }
    }

    pub fn record(&mut self) {
        self.count += 1;
    }

    pub fn rate_per_sec(&mut self) -> f64 {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_reset).as_secs_f64();
        if elapsed >= 1.0 {
            let rate = self.count as f64 / elapsed;
            self.count = 0;
            self.last_reset = now;
            rate
        } else {
            self.count as f64 / elapsed.max(0.001)
        }
    }
}

pub struct AppState {
    pub batch_map: dashmap::DashMap<batch::BatchKey, Arc<std::sync::Mutex<batch::BatchSlot>>>,
    pub http_client: Client<HttpConnector, Full<Bytes>>,
    pub rl_client: Option<Arc<AsyncMutex<RlAgentClient<Channel>>>>,
    pub telemetry: Option<Arc<crate::telemetry::TelemetryPublisher>>,
    pub cache: Option<crate::cache::ResponseCache>,
    pub latency_tracker: std::sync::Mutex<LatencyTracker>,
    pub rate_counter: std::sync::Mutex<RateCounter>,
}
