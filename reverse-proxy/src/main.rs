use std::sync::Arc;

mod listener;
mod config;
mod state;
mod service;

#[tokio::main]
async fn main(){
    //dummy config struct
    let dummy_config = Arc::new(config::Config {
        listen_addr: "127.0.0.1:8080".parse().unwrap(),
        max_connections: 10
    });

    //dummy state struct
    let dummy_state = Arc::new(state::AppState {});

    //shutdown channel
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());

    //call listener's run()
    listener::run(dummy_state, dummy_config, shutdown_rx).await.unwrap();
}
