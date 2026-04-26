use std::sync::Arc;

mod listener;
mod config;
mod state;
mod service;
mod batch;
mod router;

#[tokio::main]
async fn main(){
    //dummy config struct
    let dummy_config = Arc::new(config::Config {
        listen_addr: "127.0.0.1:8080".parse().unwrap(),
        max_connections: 10,
        batch_timeout_ms: 100,
        max_batch_size: 64,
    });

    // dummy state struct
    let dummy_state = Arc::new(state::AppState {
        batch_map: dashmap::DashMap::new(),
    });

    //shutdown channel
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());

    //call listener's run()
    listener::run(dummy_state, dummy_config, shutdown_rx).await.unwrap();
}
