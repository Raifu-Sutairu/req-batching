use crate::batch;
use std::sync::Arc;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use http_body_util::Full;
use bytes::Bytes;

pub struct AppState {
    pub batch_map: dashmap::DashMap<batch::BatchKey, Arc<std::sync::Mutex<batch::BatchSlot>>>,
    pub http_client: Client<HttpConnector, Full<Bytes>>,
}

