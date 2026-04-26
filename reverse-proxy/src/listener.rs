use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::{watch, Semaphore};
use tracing::{info, error};

//these structs will be defined in other files later
use crate::state::AppState;
use crate::config::Config;

//runs the main tcp accpt loop
pub async fn run(
    state: Arc<AppState>, 
    config: Arc<Config>,
    mut shutdown_rx: watch::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    //bind tcp socket
    let addr = &config.listen_addr;
    let listener = TcpListener::bind(addr).await?;

    info!("Listening on {}", addr);

    //create the connection limit semaphore
    let connection_limit = Arc::new(Semaphore::new(config.max_connections));

    //accpt loop
    loop {
        //wait for connection limit or a shutdown signal
        //do this before accpting a connection req
        //if we are at max -> pause here until some prev connection finishes
        let permit = tokio::select! {
            p = connection_limit.clone().acquire_owned() => {
                p.expect("Semaphore should never be closed")
            }
            _ = shutdown_rx.changed() => {
                info!("Shutdown signal while waiting for permit... Shutting down.");
                break;
            }
        };

        //wait for tcp connection or shutdown signal
        let (stream, remote_addr) = tokio::select! {
            res = listener.accept() => {
                match res{
                    Ok(conn) => conn,
                    Err(e) => {
                        error!("Failed to accept connection: {}", e);

                        //drop the permit here so that its returned to the pool
                        drop(permit);

                        //backoff briefly to avoid full usage of CPU
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        continue;
                    }
                }
            }

            _ = shutdown_rx.changed() => {
                info!("Shutdown signal recieved while waiting for connection... Shutting down.");
                break;
            }
        };

        info!("Accepted connection from {}", remote_addr);

        //spawning task and wiring hyper
        //clone the Arcs so that we can move them to the next task
        let state_clone = state.clone();
        let config_clone = config.clone();

        //clone the shutdown receiever as well
        let mut shutdown_rx_clone = shutdown_rx.clone();

        //spawning a cheap tokio task for this indvidual connection
        tokio::spawn(async move {
            //move the permit to this task (when task ends, permit is dropped)
            let _permit = permit;
            
            //wrap tokio's tcpstream in hyper's IO trait
            let io = hyper_util::rt::TokioIo::new(stream);

            //create our own custom http service
            let service = crate::service::ProxyService::new(state_clone, config_clone);

            //bind connection to hyper's HTTP/1 server
            let conn = hyper::server::conn::http1::Builder::new().serve_connection(io, service);

            //we pin the connection future because select! requires it
            tokio::pin!(conn);

            //run the connection, also listen for the shutdown signal
            tokio::select! {
                res = &mut conn => {
                    if let Err(err) = res {
                        error!("Error serving connection: {}", err);
                    }
                }

                _ = shutdown_rx_clone.changed() => {
                    info!("Shutdown signal received, removing active connection...");

                    //tells hyper to finish the current in-flight request, but reject any new pipelined requests on this socket
                    conn.as_mut().graceful_shutdown();

                    //wait for the in flight to finish
                    if let Err(err) = conn.await {
                        error!("Error during graceful shutdown: {}", err);
                    }
                }
            }
        });
    }
    
    Ok(())
}