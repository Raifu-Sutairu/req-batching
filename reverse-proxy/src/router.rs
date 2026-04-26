use crate::batch;

pub enum RoutingDecision{
    PassThrough,
    Batch(batch::BatchKey),
}

pub fn route(req: &hyper::Request<hyper::body::Incoming>) -> RoutingDecision{
    //for now we only batch GET requests
    match req.method() {
        &hyper::Method::GET => {
            let batch_key = batch::BatchKey{
                method: hyper::Method::GET,
                path: req.uri().path().to_string()
            };
            RoutingDecision::Batch(batch_key)
        },
        _ => {
            RoutingDecision::PassThrough
        }
    }
}