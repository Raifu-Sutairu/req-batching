#[derive(Debug, Clone)]
pub struct Config{
    pub listen_addr: std::net::SocketAddr,
    pub max_connections: usize //global for entire rev proxy. protects rev proxy from running out of memory
}
