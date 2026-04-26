use std::sync::Arc;

use crate::state::AppState;
use crate::config::Config;

//gomma hyper MIGHT complain about String in Response<String>
pub async fn handle_request(
    req: hyper::Request<hyper::body::Incoming>,
    state: Arc<AppState>,
    config: Arc<Config>
) -> Result<hyper::Response<String>, std::convert::Infallible> {
    //hardcoded 200 response
    Ok(hyper::Response::builder().status(200).body("Hello from the proxy".to_string()).unwrap())
}