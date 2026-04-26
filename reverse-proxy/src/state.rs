use crate::batch;
use std::sync::Arc;

pub struct AppState{
    //using tokio's mutex here will be slower imo, as pure memory ops -> use the std lib mutex
    pub batch_map: dashmap::DashMap<batch::BatchKey, Arc<std::sync::Mutex<batch::BatchSlot>>>,
}

