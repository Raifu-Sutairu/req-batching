use std::sync::Arc;

mod listener;
mod config;
mod state;
mod service;
mod batch;
mod router;

#[tokio::main]
async fn main() {
    //initialize tracing logs
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("reverse_proxy=info".parse().unwrap()),
        )
        .init();

    //load configuration
    let config = Arc::new(config::Config::load().expect("Failed to load configuration"));
    tracing::info!("Configuration loaded: {:?}", config);

    let http_client = hyper_util::client::legacy::Client::builder(
        hyper_util::rt::TokioExecutor::new(),
    )
    .build_http();

    //dummy state struct
    let app_state = Arc::new(state::AppState {
        batch_map: dashmap::DashMap::new(),
        http_client,
    });

    //shutdown channel
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());

    //call listener's run()
    listener::run(app_state, config, shutdown_rx).await.unwrap();
}
