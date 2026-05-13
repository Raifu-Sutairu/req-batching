use std::sync::Arc;
use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::Request;
use hyper::body::Incoming;

use crate::state::AppState;
use crate::batch::{BatchSlot, BatchState, BatchKey};
use crate::config::Config;
use crate::router;

pub async fn handle_request(
    mut req: Request<Incoming>,
    state: Arc<AppState>,
    config: Arc<Config>
) -> Result<hyper::Response<http_body_util::Full<Bytes>>, std::convert::Infallible> {
    // Record incoming request for rate tracking
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

            let (is_first, should_serve_now) = {
                let mut slot = slot_arc.lock().unwrap();
                slot.senders.push(tx);
                (slot.senders.len() == 1, slot.senders.len() >= config.max_batch_size)
            };

            let state_clone = state.clone();
            let config_clone = config.clone();
            
            if is_first {
                let batch_slot_clone = slot_arc.clone();
                let batch_key_clone = key.clone();
                
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(config_clone.batch_timeout_ms)).await;
                    serve_batch(batch_slot_clone, batch_key_clone, state_clone, config_clone, "Timer expired").await;
                });
            }

            if should_serve_now {
                let batch_slot_clone = slot_arc.clone();
                let batch_key_clone = key.clone();
                let state_clone2 = state.clone();
                let config_clone2 = config.clone();
                tokio::spawn(async move {
                    serve_batch(batch_slot_clone, batch_key_clone, state_clone2, config_clone2, "Size limit reached").await;
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
    batch_slot: Arc<std::sync::Mutex<BatchSlot>>,
    batch_key: BatchKey,
    state: Arc<AppState>,
    config: Arc<Config>,
    reason: &str, 
) {
    let (batch_size, batch_age_ms) = {
        let locked = batch_slot.lock().unwrap();
        if let BatchState::Serving = locked.state {
            return;
        }
        (locked.senders.len() as u32, locked.created_at.elapsed().as_millis() as f32)
    };

    let p99 = state.latency_tracker.lock().unwrap().p99() as f32;
    let rate = state.rate_counter.lock().unwrap().rate_per_sec() as f32;

    let mut decision_reason = reason.to_string();

    // Query RL Agent
    if let Some(rl_client_arc) = &state.rl_client {
        let agent_call = async {
            let mut client = rl_client_arc.lock().await;
            let req = tonic::Request::new(crate::proto::rl_agent::BatchState {
                batch_size,
                batch_age_ms,
                upstream_p99_ms: p99,
                request_rate: rate,
            });
            client.decide(req).await
        };

        match tokio::time::timeout(std::time::Duration::from_millis(config.rl_agent_timeout_ms), agent_call).await {
            Ok(Ok(response)) => {
                let decision = response.into_inner();
                if !decision.should_flush {
                    tracing::info!(batch_size = batch_size, batch_age_ms = batch_age_ms, "RL Agent decided to WAIT. Deferring flush.");
                    return;
                }
                decision_reason = "RL Agent".to_string();
            },
            Ok(Err(e)) => {
                tracing::warn!("RL Agent gRPC error: {}. Falling back to heuristics.", e);
            },
            Err(_) => {
                tracing::warn!("RL Agent timeout. Falling back to heuristics.");
            }
        }
    }

    let senders = {
        let mut locked = batch_slot.lock().unwrap();
        match locked.state {
            BatchState::Waiting => {
                locked.state = BatchState::Serving;
                state.batch_map.remove(&batch_key);
                std::mem::take(&mut locked.senders)
            },
            BatchState::Serving => return, //already serving
        }
    }; //lock dropped

    tracing::info!(reason = %decision_reason, batch_size = senders.len(), "Batch flushed");

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
        .method(batch_key.method)
        .uri(uri)
        .body(http_body_util::Full::new(Bytes::new()))
        .unwrap();

    let start = std::time::Instant::now();
    let response = match state.http_client.request(req).await {
        Ok(resp) => {
            state.latency_tracker.lock().unwrap().record(start.elapsed().as_secs_f64() * 1000.0);
            let (parts, body) = resp.into_parts();
            let body_bytes = match body.collect().await {
                Ok(c) => c.to_bytes(),
                Err(_) => Bytes::new()
            };
            hyper::Response::from_parts(parts, http_body_util::Full::new(body_bytes))
        },
        Err(e) => {
            tracing::error!("Batch upstream error: {:?}", e);
            hyper::Response::builder()
                .status(502)
                .body(http_body_util::Full::new(Bytes::from("upstream error")))
                .unwrap()
        }
    };

    //clone the response for all senders
    let (parts, body) = response.into_parts();
    for tx in senders {
        let mut cloned_parts = hyper::Response::builder()
            .status(parts.status.clone())
            .version(parts.version.clone());
        for (k, v) in parts.headers.iter() {
            cloned_parts = cloned_parts.header(k.clone(), v.clone());
        }
        let cloned_resp = cloned_parts.body(body.clone()).unwrap();
        let _ = tx.send(cloned_resp);
    }
}