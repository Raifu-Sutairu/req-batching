use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub listen_addr: std::net::SocketAddr,
    pub upstream_url: String,
    pub max_connections: usize,
    pub batch_timeout_ms: u64,
    pub max_batch_size: usize,
}

impl Config {
    pub fn load() -> Result<Self, config::ConfigError> {
        config::Config::builder()
            .add_source(config::File::with_name("config").required(false))
            .add_source(config::Environment::with_prefix("PROXY"))
            .build()?
            .try_deserialize()
    }
}
