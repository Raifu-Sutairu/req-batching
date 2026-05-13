use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use axum::{routing::get, Router, extract::State};

#[tokio::main]
async fn main() {
    //shared hit counter
    let hit_counter = Arc::new(AtomicUsize::new(0));

    let app = Router::new()
        .route(
            "/{*path}",
            get(handle_request),
        )
        .with_state(hit_counter);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9090").await.unwrap();
    println!("Echo server listening on 0.0.0.0:9090");
    axum::serve(listener, app).await.unwrap();
}

async fn handle_request(State(counter): State<Arc<AtomicUsize>>) -> String {
    //increment and get the new value
    let hits = counter.fetch_add(1, Ordering::SeqCst) + 1;
    println!("Echo server received a request! Total upstream hits: {}", hits);
    
    format!("Hello from the echo server! Your request was batched or proxied. (Total Upstream Hits: {})", hits)
}
