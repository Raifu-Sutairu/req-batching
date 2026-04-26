use std::sync::Arc;

use crate::state::AppState;
use crate::batch::{BatchSlot, BatchState};
use crate::config::Config;
use crate::router;

//gomma hyper MIGHT complain about String in Response<String>
pub async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    state: Arc<AppState>,
    config: Arc<Config>
) -> Result<hyper::Response<String>, std::convert::Infallible> {
    let decision = router::route(&req);

    match decision {
        router::RoutingDecision::PassThrough => {
            Ok(hyper::Response::builder().status(200).body("Passed through directly".to_string()).unwrap())
        },
        router::RoutingDecision::Batch(key) => {
            let (tx, rx) = tokio::sync::oneshot::channel();

            //entry api is safe
            let entry = state.batch_map.entry(key).or_insert_with(|| {
                Arc::new(std::sync::Mutex::new(BatchSlot{
                        state: BatchState::Waiting,
                        created_at: std::time::Instant::now(),
                        senders: Vec::new(),
                }))
            });

            entry.lock().unwrap().senders.push(tx);

            let response = rx.await.unwrap();
            Ok(response)
        },
        _ => {
            Ok(hyper::Response::builder().status(500).body("Internal server error".to_string()).unwrap())
        }
    }
}