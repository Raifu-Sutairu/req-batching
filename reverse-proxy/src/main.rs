use std::sync::Arc;

mod listener;
mod config;
mod state;
mod proto;
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

    //try to connect to the rl agent
    let rl_client = if config.rl_agent_enabled {
        match crate::proto::rl_agent::rl_agent_client::RlAgentClient::connect(config.rl_agent_addr.clone()).await {
            Ok(client) => {
                tracing::info!("Successfully connected to RL Agent");
                Some(Arc::new(tokio::sync::Mutex::new(client)))
            },
            Err(e) => {
                tracing::warn!("Failed to connect to RL Agent: {}. Falling back to heuristics.", e);
                None
            }
        }
    } else {
        None
    };

    //dummy state struct
    let app_state = Arc::new(state::AppState {
        batch_map: dashmap::DashMap::new(),
        http_client,
        rl_client,
        latency_tracker: std::sync::Mutex::new(state::LatencyTracker::new(1000)),
        rate_counter: std::sync::Mutex::new(state::RateCounter::new()),
    });

    //shutdown channel
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());

    //call listener's run()
    listener::run(app_state, config, shutdown_rx).await.unwrap();
}
