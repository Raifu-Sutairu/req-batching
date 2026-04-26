use std::sync::Arc;

use crate::state::AppState;
use crate::batch::{BatchSlot, BatchState, BatchKey};
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
            let entry = state.batch_map.entry(key.clone()).or_insert_with(|| {
                Arc::new(std::sync::Mutex::new(BatchSlot{
                        state: BatchState::Waiting,
                        created_at: std::time::Instant::now(),
                        senders: Vec::new(),
                }))
            });

            //need to detect the first request to start the timer and the batch size
            let (is_first, should_serve_now) = {
                let mut slot = entry.lock().unwrap();
                slot.senders.push(tx);

                (slot.senders.len() == 1, slot.senders.len() >= config.max_batch_size)
            };

            //if is_first is TRUE -> spawn a new task!
            if is_first {
                let batch_slot_clone = entry.value().clone();
                let batch_key_clone = key.clone();
                let state_batch_map_clone = state.batch_map.clone();
                let config_batch_timeout = config.batch_timeout_ms;

                //timer task
                tokio::spawn(async move {
                    //sleep for max wait time
                    tokio::time::sleep(std::time::Duration::from_millis(config_batch_timeout)).await;

                    //helper function call
                    serve_batch(batch_slot_clone, &batch_key_clone, &state_batch_map_clone, "Timer expired");
                });
            }

            if should_serve_now {
                serve_batch(entry.value().clone(), &key, &state.batch_map, "Size limit reached");
            }

            let response = rx.await.unwrap();
            Ok(response)
        },
    }
}

//helper funcs
fn serve_batch(
    batch_slot: Arc<std::sync::Mutex<BatchSlot>>,
    batch_key: &BatchKey,
    batch_map: &dashmap::DashMap<BatchKey, Arc<std::sync::Mutex<BatchSlot>>>,
    reason: &str, 
) {
    //wake up, lock the slot
    let mut locked = batch_slot.lock().unwrap();

    match locked.state{
        BatchState::Waiting => {
            locked.state = BatchState::Serving;

            //remove the map entry so no new reqs can join this batch
            batch_map.remove(&batch_key);

            //steal the array of senders out of the lock safely
            let senders = std::mem::take(&mut locked.senders);

            //we no longer need the lock, so manually drop it to free it up early
            drop(locked);

            //loop through all the waiting clients and send them the response!
            for tx in senders {
                let response = hyper::Response::builder().status(200).body("Mock batched response from the Timer!".to_string()).unwrap();

                //we use _, bcoz if client disconnected while waiting, tx will return err. We dont care if they disconn - so ignore err!
                let _ = tx.send(response);
            }

            println!("Forcing the batch to serve due to {}", reason);
        },
        BatchState::Serving => {
            //RL agent wouldve served this batch
            //quitely exit
        }
    }
}