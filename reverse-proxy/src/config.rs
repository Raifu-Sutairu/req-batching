#[derive(Debug, Clone)]
pub struct Config{
    pub listen_addr: std::net::SocketAddr,
    pub max_connections: usize
}
