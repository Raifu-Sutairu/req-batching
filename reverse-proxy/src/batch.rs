use hyper::http;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct BatchKey{
    pub method: http::Method,
    pub path: String
}

pub enum BatchState{
    Waiting,
    Serving,
}

//slots
pub struct BatchSlot{
    state: BatchState,
    created_at: std::time::Instant,
    senders: Vec<tokio::sync::oneshot::Sender<hyper::Response<String>>>
}