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

            match state.http_client.request(req_full).await {
                Ok(resp) => {
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

    tracing::info!(reason = %reason, "Batch flushed");

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

    let response = match state.http_client.request(req).await {
        Ok(resp) => {
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