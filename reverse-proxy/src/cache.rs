use redis::AsyncCommands;
use bytes::Bytes;

#[derive(Clone)]
pub struct ResponseCache {
    client: redis::aio::ConnectionManager,
    ttl_seconds: u64,
}

impl ResponseCache {
    pub async fn new(url: &str, ttl_seconds: u64) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(url)?;
        let connection_manager = client.get_connection_manager().await?;
        
        Ok(Self {
            client: connection_manager,
            ttl_seconds,
        })
    }

    pub async fn get(&self, key: &str) -> Option<Bytes> {
        let mut conn = self.client.clone();
        match conn.get::<_, Option<Vec<u8>>>(key).await {
            Ok(Some(data)) => {
                tracing::debug!("Cache hit for key: {}", key);
                Some(Bytes::from(data))
            },
            Ok(None) => {
                tracing::debug!("Cache miss for key: {}", key);
                None
            },
            Err(e) => {
                tracing::warn!("Redis GET error for key {}: {}", key, e);
                None
            }
        }
    }

    pub async fn set(&self, key: &str, body: Bytes) {
        let mut conn = self.client.clone();
        let data = body.to_vec();
        
        match conn.set_ex::<_, _, ()>(key, data, self.ttl_seconds).await {
            Ok(_) => tracing::debug!("Cached response for key: {} (TTL: {}s)", key, self.ttl_seconds),
            Err(e) => tracing::warn!("Redis SET error for key {}: {}", key, e),
        }
    }
}
