use std::sync::Arc;

mod listener;
mod config;
mod state;
mod proto;
mod service;
mod batch;
mod router;

mod telemetry;
mod cache;

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

    //try to connect to the rl agent with retries
    let rl_client = if config.rl_agent_enabled {
        let mut client_opt = None;
        for i in 1..=5 {
            match crate::proto::rl_agent::rl_agent_client::RlAgentClient::connect(config.rl_agent_addr.clone()).await {
                Ok(client) => {
                    tracing::info!("Successfully connected to RL Agent");
                    client_opt = Some(Arc::new(tokio::sync::Mutex::new(client)));
                    break;
                },
                Err(e) => {
                    tracing::warn!("Failed to connect to RL Agent (attempt {}/5): {}. Retrying in 1s...", i, e);
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
        if client_opt.is_none() {
            tracing::warn!("Could not connect to RL Agent after 5 attempts. Falling back to heuristics.");
        }
        client_opt
    } else {
        None
    };

    //initialize telemetry publisher
    let telemetry = if config.kafka_enabled {
        match telemetry::TelemetryPublisher::new(&config.kafka_brokers, &config.kafka_telemetry_topic) {
            Ok(publisher) => {
                tracing::info!("Kafka telemetry publisher initialized");
                Some(Arc::new(publisher))
            },
            Err(e) => {
                tracing::warn!("Failed to initialize Kafka telemetry: {}. Telemetry will be disabled.", e);
                None
            }
        }
    } else {
        None
    };

    //initialize redis cache
    let cache = if config.redis_enabled {
        match cache::ResponseCache::new(&config.redis_url, config.redis_cache_ttl).await {
            Ok(cache) => {
                tracing::info!("Redis cache initialized");
                Some(cache)
            },
            Err(e) => {
                tracing::warn!("Failed to initialize Redis cache: {}. Caching will be disabled.", e);
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
        telemetry,
        cache,
        latency_tracker: std::sync::Mutex::new(state::LatencyTracker::new(1000)),
        rate_counter: std::sync::Mutex::new(state::RateCounter::new()),
    });

    //shutdown channel
    let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(());

    //call listener's run()
    listener::run(app_state, config, shutdown_rx).await.unwrap();
}
