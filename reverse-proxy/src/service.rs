use std::sync::Arc;
use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::Request;
use hyper::body::Incoming;

use crate::state::AppState;
use crate::batch::{BatchSlot, BatchState, BatchKey};
use crate::config::Config;
use crate::router;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::telemetry::BatchFlushEvent;

#[derive(Debug, Clone)]
pub enum FlushReason {
    SizeCap,
    Timeout,
    RlAgent,
    RlAgentTimeout,
    RlAgentError,
}

impl FlushReason {
    pub fn to_str(&self) -> &'static str {
        match self {
            FlushReason::SizeCap => "SizeCap",
            FlushReason::Timeout => "Timeout",
            FlushReason::RlAgent => "RlAgent",
            FlushReason::RlAgentTimeout => "RlAgentTimeout",
            FlushReason::RlAgentError => "RlAgentError",
        }
    }
}

pub async fn handle_request(
    mut req: Request<Incoming>,
    state: Arc<AppState>,
    config: Arc<Config>
) -> Result<hyper::Response<http_body_util::Full<Bytes>>, std::convert::Infallible> {
    //record incoming request for rate tracking
    state.rate_counter.lock().unwrap().record();

    let decision = router::route(&req);

    match decision {
        router::RoutingDecision::PassThrough => {
            //forward request directly
            let uri_string = format!("{}{}", config.upstream_url, req.uri().path_and_query().map(|x| x.as_str()).unwrap_or(""));
            let uri: hyper::Uri = match uri_string.parse() {
                Ok(u) => u,
                Err(e) => {
                    tracing::error!("Failed to parse PassThrough URI: {:?}", e);
                    return Ok(hyper::Response::builder()
                        .status(502)
                        .body(http_body_util::Full::new(Bytes::from("upstream error")))
                        .unwrap());
                }
            };
            
            *req.uri_mut() = uri;
            
            //req is currently Request<Incoming>. We need to convert it to Request<Full<Bytes>> to pass to http_client
            let (parts, body) = req.into_parts();
            let body_bytes = match body.collect().await {
                Ok(c) => c.to_bytes(),
                Err(e) => {
                    tracing::error!("Failed to read request body: {:?}", e);
                    Bytes::new()
                }
            };
            let req_full = Request::from_parts(parts, http_body_util::Full::new(body_bytes));

            let start = std::time::Instant::now();
            match state.http_client.request(req_full).await {
                Ok(resp) => {
                    state.latency_tracker.lock().unwrap().record(start.elapsed().as_secs_f64() * 1000.0);
                    let (parts, body) = resp.into_parts();
                    let body_bytes = match body.collect().await {
                        Ok(c) => c.to_bytes(),
                        Err(_) => Bytes::new()
                    };
                    Ok(hyper::Response::from_parts(parts, http_body_util::Full::new(body_bytes)))
                },
                Err(e) => {
                    tracing::error!("PassThrough upstream error: {:?}", e);
                    Ok(hyper::Response::builder()
                        .status(502)
                        .body(http_body_util::Full::new(Bytes::from("upstream error")))
                        .unwrap())
                }
            }
        },
        router::RoutingDecision::Batch(key) => {
            let (tx, rx) = tokio::sync::oneshot::channel();

            let slot_arc = {
                let entry = state.batch_map.entry(key.clone()).or_insert_with(|| {
                    Arc::new(std::sync::Mutex::new(BatchSlot{
                            state: BatchState::Waiting,
                            created_at: std::time::Instant::now(),
                            senders: Vec::new(),
                    }))
                });
                entry.value().clone()
            }; //entry is dropped here, releasing the DashMap shard lock!

            let (is_first, batch_size, batch_age_ms) = {
                let mut slot = slot_arc.lock().unwrap();
                slot.senders.push(tx);
                (slot.senders.len() == 1, slot.senders.len(), slot.created_at.elapsed().as_secs_f32() * 1000.0)
            };

            if is_first {
                let state_clone = Arc::clone(&state);
                let key_clone = key.clone();
                let timeout = config.batch_timeout_ms;
                let config_clone = Arc::clone(&config);
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(timeout)).await;
                    serve_batch(key_clone, state_clone, config_clone, FlushReason::Timeout).await;
                });
            }

            // Determine if we should flush
            let mut reason = None;
            
            // HARD LIMIT 1: size cap
            if batch_size >= config.max_batch_size {
                reason = Some(FlushReason::SizeCap);
            } 
            // HARD LIMIT 2: timeout cap
            else if batch_age_ms >= config.batch_timeout_ms as f32 {
                reason = Some(FlushReason::Timeout);
            } 
            // SOFT DECISION: Ask RL agent
            else {
                let p99 = state.latency_tracker.lock().unwrap().p99() as f32;
                let rate = state.rate_counter.lock().unwrap().rate_per_sec() as f32;

                if let Some(rl_client_arc) = &state.rl_client {
                    tracing::debug!(
                        batch_age_ms = batch_age_ms,
                        batch_size = batch_size,
                        "Querying RL agent"
                    );
                    let agent_call = async {
                        let mut client = rl_client_arc.lock().await;
                        let req = tonic::Request::new(crate::proto::rl_agent::BatchState {
                            batch_size: batch_size as u32,
                            batch_age_ms,
                            upstream_p99_ms: p99,
                            request_rate: rate,
                        });
                        client.decide(req).await
                    };

                    match tokio::time::timeout(std::time::Duration::from_millis(config.rl_agent_timeout_ms), agent_call).await {
                        Ok(Ok(response)) => {
                            let decision = response.into_inner();
                            if decision.should_flush {
                                reason = Some(FlushReason::RlAgent);
                            }
                        },
                        Ok(Err(e)) => {
                            tracing::warn!("RL Agent gRPC error: {}. Falling back to heuristics.", e);
                            // We don't flush immediately, we let heuristics take over when limits are reached
                        },
                        Err(_) => {
                            tracing::warn!("RL Agent timeout. Falling back to heuristics.");
                        }
                    }
                }
            }

            if let Some(r) = reason {
                let key_clone = key.clone();
                let state_clone = Arc::clone(&state);
                let config_clone = Arc::clone(&config);
                tokio::spawn(async move {
                    serve_batch(key_clone, state_clone, config_clone, r).await;
                });
            }

            let response = rx.await.unwrap_or_else(|_| {
                hyper::Response::builder()
                    .status(502)
                    .body(http_body_util::Full::new(Bytes::from("upstream error")))
                    .unwrap()
            });
            
            Ok(response)
        },
    }
}

async fn serve_batch(
    batch_key: BatchKey,
    state: Arc<AppState>,
    config: Arc<Config>,
    reason: FlushReason, 
) {
    let slot_arc = match state.batch_map.get(&batch_key) {
        Some(entry) => entry.value().clone(),
        None => return, // already flushed
    };

    let senders = {
        let mut locked = slot_arc.lock().unwrap();
        match locked.state {
            BatchState::Waiting => {
                locked.state = BatchState::Serving;
                state.batch_map.remove(&batch_key);
                std::mem::take(&mut locked.senders)
            },
            BatchState::Serving => return, // already serving, idempotency check
        }
    }; // lock dropped

    let batch_size = senders.len();
    // Recompute age at flush time
    let batch_age_ms = locked_created_at_elapsed(&slot_arc);

    let reason_str = reason.to_str();
    if matches!(reason, FlushReason::Timeout) {
        tracing::info!(
            reason = %reason_str,
            batch_size = batch_size,
            "Heuristic timer fired — flushing batch"
        );
    } else {
        tracing::info!(reason = %reason_str, batch_size = batch_size, "Batch flushed");
    }

    let cache_key = format!("{}:{}", batch_key.method, batch_key.path);
    let mut cache_hit = false;
    let mut response_body = Bytes::new();
    let mut status_code = hyper::StatusCode::OK;

    if let Some(cache) = &state.cache {
        if let Some(cached_bytes) = cache.get(&cache_key).await {
            cache_hit = true;
            response_body = cached_bytes;
        }
    }

    let start = std::time::Instant::now();

    if !cache_hit {
        let uri_string = format!("{}{}", config.upstream_url, batch_key.path);
        let uri: hyper::Uri = match uri_string.parse() {
            Ok(u) => u,
            Err(_) => {
                tracing::error!("Failed to parse upstream URI");
                for tx in senders {
                    let _ = tx.send(hyper::Response::builder().status(502).body(http_body_util::Full::new(Bytes::from("upstream error"))).unwrap());
                }
                return;
            }
        };

        let req = Request::builder()
            .method(batch_key.method.clone())
            .uri(uri)
            .body(http_body_util::Full::new(Bytes::new()))
            .unwrap();

        match state.http_client.request(req).await {
            Ok(resp) => {
                state.latency_tracker.lock().unwrap().record(start.elapsed().as_secs_f64() * 1000.0);
                let (parts, body) = resp.into_parts();
                status_code = parts.status;
                response_body = match body.collect().await {
                    Ok(c) => c.to_bytes(),
                    Err(_) => Bytes::new()
                };

                if status_code == hyper::StatusCode::OK {
                    if let Some(cache) = &state.cache {
                        cache.set(&cache_key, response_body.clone()).await;
                    }
                }
            },
            Err(e) => {
                tracing::error!("Batch upstream error: {:?}", e);
                status_code = hyper::StatusCode::BAD_GATEWAY;
                response_body = Bytes::from("upstream error");
            }
        };
    }

    let upstream_ms = start.elapsed().as_secs_f64() * 1000.0;

    for tx in senders {
        let cloned_resp = hyper::Response::builder()
            .status(status_code)
            .body(http_body_util::Full::new(response_body.clone()))
            .unwrap();
        let _ = tx.send(cloned_resp);
    }

    // emit telemetry
    if let Some(telemetry) = &state.telemetry {
        let timestamp_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let event = crate::telemetry::BatchFlushEvent {
            batch_key: format!("{}:{}", batch_key.method, batch_key.path),
            batch_size: batch_size as usize,
            batch_age_ms: batch_age_ms as f64,
            upstream_ms,
            flush_reason: reason_str.to_string(),
            timestamp_unix,
        };

        telemetry.publish_flush(event).await;
    }
}

fn locked_created_at_elapsed(slot_arc: &Arc<std::sync::Mutex<BatchSlot>>) -> f32 {
    let locked = slot_arc.lock().unwrap();
    locked.created_at.elapsed().as_secs_f32() * 1000.0
}