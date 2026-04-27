# Request Batch Dispatching with RL-Based Adaptive Flushing

A production-ready, high-performance reverse proxy written in Rust that groups concurrent HTTP requests into batches and dispatches them as a single upstream call. The flush decision — when to serve a batch — is designed to be driven by a Reinforcement Learning agent trained on live traffic state, replacing naive fixed timers with a learned optimal policy.

---

## Overview

Modern backend services receive bursts of identical or near-identical requests within short time windows. Forwarding each one independently wastes upstream capacity and inflates latency under load. This proxy intercepts those requests, holds them in a shared batch slot, and flushes the entire group with a single upstream call, fanning the single response back out to every waiting client.

The core differentiator is the flush policy. Rather than a hardcoded timer or size cap, the system is architected to query an RL agent (PPO-based) over gRPC. The agent observes real-time state — queue depth, arrival rate, batch age, upstream latency — and returns a binary WAIT or SERVE decision. This allows the proxy to adapt its batching behavior to traffic patterns it has never seen before without redeployment or config changes.

---

## Status

The core proxy engine is complete and functional. All batching primitives, the TCP listener, the HTTP service layer, the router, and the batch state machine are implemented and ready for integration. The RL agent gRPC interface and upstream HTTP forwarding are the remaining integration points.

| Component | Status |
|---|---|
| TCP listener with connection cap | Complete |
| Graceful shutdown via watch channel | Complete |
| HTTP/1 service layer (Hyper) | Complete |
| Request router (Batch vs PassThrough) | Complete |
| Concurrent batch registry (DashMap) | Complete |
| BatchSlot state machine | Complete |
| Dual flush triggers (timer + size cap) | Complete |
| Idempotent serve_batch with state guard | Complete |
| Static TOML configuration struct | Complete |
| Arc-shared AppState across tasks | Complete |
| Upstream HTTP forwarding | Integration pending |
| RL agent gRPC client (Tonic) | Integration pending |
| Redis dedup cache | Integration pending |
| Kafka telemetry / replay buffer | Integration pending |

---

## Capabilities

### TCP Listener

- Binds to any configurable `SocketAddr` via `TcpListener`
- Non-blocking accept loop using `tokio::select!` to interleave acceptance with shutdown signals
- Semaphore-based global connection cap — pauses new accepts when the limit is reached, preventing memory exhaustion under traffic spikes
- Per-connection backoff on accept errors to avoid CPU spin
- Each accepted stream is handed off to an isolated `tokio::spawn` task

### Graceful Shutdown

- Shutdown is coordinated via a `tokio::sync::watch` channel broadcast to every active connection task
- On signal, the listener stops accepting new connections
- Each in-flight connection completes its current request via `hyper::server::conn::http1::Builder::graceful_shutdown()` before the task exits
- The semaphore permit is tied to the task lifetime — automatically returned when the task drops

### HTTP/1 Service Layer

- Each TCP stream is wrapped with `hyper_util::rt::TokioIo` and served by a Hyper HTTP/1 connection
- Requests are dispatched to a `service_fn` closure that carries cloned `Arc<AppState>` and `Arc<Config>` references — zero contention on the hot path
- The connection future is raced against the shutdown receiver using `tokio::select!`

### Request Router

- Classifies every incoming request into one of two decisions:
  - `RoutingDecision::Batch(BatchKey)` — for `GET` requests, keyed on `(method, path)`
  - `RoutingDecision::PassThrough` — for all other HTTP methods
- The router is a pure function with no shared state — trivial to extend with path-prefix rules, header inspection, or tenant-aware routing
- `BatchKey` is `Hash + Eq` — used directly as a `DashMap` key

### Concurrent Batch Registry

- Global registry: `DashMap<BatchKey, Arc<Mutex<BatchSlot>>>`
- `DashMap` provides shard-level locking for concurrent access across Tokio tasks — no single global mutex
- The inner `std::sync::Mutex` (not Tokio's) is intentional: batch slot operations are pure memory ops with no async yield points, making `std::sync::Mutex` faster and free of async deadlock risk
- The `entry().or_insert_with()` API is atomic — only one task ever initializes a slot for a given key

### BatchSlot and State Machine

- Each `BatchSlot` holds:
  - `state: BatchState` — either `Waiting` or `Serving`
  - `created_at: Instant` — timestamp of the first request in the batch (feeds into RL state vector)
  - `senders: Vec<oneshot::Sender<Response<String>>>` — one per parked request
- Incoming requests push their `oneshot::Sender` into the slot, then `await` the `Receiver` — the task is suspended at zero CPU cost until the batch is served

### Dual Flush Triggers

Two independent paths can trigger a batch flush, whichever fires first:

1. **Timer trigger** — the first request into a slot spawns a background Tokio task that sleeps for `batch_timeout_ms` milliseconds, then calls `serve_batch()`. This guarantees maximum latency even under low arrival rates.
2. **Size cap trigger** — after each push, if `senders.len() >= max_batch_size`, `serve_batch()` is called inline on the request task. This caps memory usage and prevents unbounded growth under bursts.

### Idempotent serve_batch

`serve_batch()` is safe to call from both the timer task and the size-cap path concurrently:

- The first caller atomically transitions the slot from `Waiting` to `Serving`
- The second caller finds `Serving` and exits immediately without double-flushing
- The slot is removed from the `DashMap` before senders are drained — no new requests can join a batch that is being served
- `std::mem::take` drains the sender vector without cloning, and the `Mutex` guard is dropped before the fan-out loop, minimizing lock hold time

### Response Fan-out

- A single upstream response is broadcast to all N waiting clients via their `oneshot::Sender`
- Disconnected clients (sender returns `Err`) are silently ignored — a dropped receiver is a normal condition when a client times out
- Each sender receives an independent `Response<String>` constructed from the upstream body

### Configuration

All tunable parameters are in `Config`:

| Field | Type | Description |
|---|---|---|
| `listen_addr` | `SocketAddr` | Address and port the proxy binds to |
| `max_connections` | `usize` | Maximum concurrent TCP connections (semaphore cap) |
| `batch_timeout_ms` | `u64` | Maximum milliseconds to hold a batch open before flushing |
| `max_batch_size` | `usize` | Maximum requests per batch before flushing immediately |

---

## Architecture

```
config.rs
  TOML config
  listen_addr, timeouts, upstream URL
          |
          v
state.rs - AppState
  Arc-wrapped shared resources
  DashMap<BatchKey, Arc<Mutex<BatchSlot>>>
          |
    ______|______
   |             |
listener.rs   (future: grpc/agent_client.rs)
  TcpListener   Tonic gRPC client
  Accept loop   Async with timeout fallback
  Semaphore
  Shutdown watch
          |
   TcpStream to Hyper
          |
          v
service.rs - ProxyService
  Hyper HTTP parsing
  Arc<AppState> + Arc<Config> cloned per connection
          |
          v
router.rs
  Endpoint fingerprint hash (method + path)
  RoutingDecision::Batch | PassThrough
          |
   push to slot
          |
          v
batch.rs - BatchSlot
  Vec of oneshot Senders
  First arrival timestamp (created_at)
          |
        trigger (timer or size cap)
          |
          v
serve_batch()
  Waiting -> Serving state transition (idempotent)
  Remove from DashMap
  mem::take senders
  Fan out response to N receivers
          |
          v
Client receives HTTP response
```

---

## Project Layout

```
req-batching/
├── reverse-proxy/          # Core proxy engine (Rust, production-ready)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Entry point: wires Config, AppState, listener
│       ├── config.rs       # Static configuration struct
│       ├── state.rs        # Shared AppState (DashMap batch registry)
│       ├── batch.rs        # BatchKey, BatchState, BatchSlot types
│       ├── router.rs       # Routing decision: Batch vs PassThrough
│       ├── listener.rs     # TCP accept loop, semaphore, graceful shutdown
│       └── service.rs      # Hyper HTTP handler, batching engine, serve_batch()
├── rl/                     # RL agent (PPO, Gymnasium env, gRPC server)
└── docs/                   # Design notes and architecture diagrams
```

---

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| `tokio` | 1.52 | Async runtime with full features (net, time, sync, macros) |
| `hyper` | 1.9 | HTTP/1 server and request/response types |
| `hyper-util` | 0.1 | `TokioIo` adapter for wrapping Tokio streams |
| `dashmap` | 6.1 | Concurrent shard-locked hash map for the batch registry |
| `tracing` | 0.1 | Structured logging throughout the proxy |

---

## Getting Started

### Prerequisites

- Rust 1.80 or later (edition 2024)
- Cargo

Install Rust via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Clone the Repository

```bash
git clone https://github.com/your-username/req-batching.git
cd req-batching
```

### Build

```bash
cd reverse-proxy
cargo build --release
```

### Run

```bash
cd reverse-proxy
cargo run
# Proxy starts and listens on 127.0.0.1:8080
```

### Test Batched Requests

Send multiple concurrent GET requests to the same endpoint. The proxy will hold them until either the batch timeout fires or the batch size cap is reached, then flush all of them with a single response.

```bash
# Send 5 concurrent GET requests to the same path
for i in {1..5}; do
  curl -s http://127.0.0.1:8080/api/data &
done
wait
```

All 5 clients will receive the same response at roughly the same time after the batch timeout elapses.

### Test PassThrough

Non-GET requests bypass the batching engine entirely and are forwarded immediately:

```bash
curl -X POST http://127.0.0.1:8080/api/submit \
     -H "Content-Type: application/json" \
     -d '{"key": "value"}'
```

### Tuning Configuration

Edit the values in `main.rs` (or a future TOML config file) to tune behavior:

```rust
let config = Arc::new(config::Config {
    listen_addr: "127.0.0.1:8080".parse().unwrap(),
    max_connections: 1000,      // raise for high-traffic deployments
    batch_timeout_ms: 50,       // lower for latency-sensitive workloads
    max_batch_size: 128,        // raise for throughput-optimized workloads
});
```

---

## Design Principles

**Protocol-agnostic batching core.** The batching engine (`batch.rs`, `state.rs`, `service.rs`) is completely decoupled from the upstream transport. Adding gRPC dispatch, a Kafka producer, or a Redis cache hit is a matter of plugging in a new `serve_batch` variant — the slot and fan-out machinery stays unchanged.

**RL-driven flush policy.** The hardcoded `batch_timeout_ms` and `max_batch_size` triggers are fallback guards. The intended production path is a gRPC call to a PPO agent that observes a state vector — queue depth, arrival rate, batch age, upstream p99 latency — and returns WAIT or SERVE. The agent learns an optimal policy from live traffic without configuration changes.

**Minimal lock contention.** `DashMap` shards the batch registry across 16 buckets by default, so concurrent tasks for different endpoints never contend. The inner `std::sync::Mutex` is held only for the duration of a `Vec::push` or `mem::take` — microseconds — before being explicitly dropped ahead of any async work.

**Zero-copy fan-out.** `std::mem::take` drains the sender vector in place. The `Arc<Mutex<BatchSlot>>` is shared by reference across the timer task and the request task — the slot data is never duplicated.

**Bounded memory.** The semaphore cap (`max_connections`) prevents unbounded TCP connection growth. The `max_batch_size` cap bounds the size of any single `BatchSlot`. Together they establish hard memory ceilings under adversarial traffic.

---

## Production Deployment

This proxy is designed as a drop-in middleware layer. It sits between any load balancer and any upstream HTTP server without requiring changes to either side. The upstream is unaware that requests were batched — it receives a single well-formed HTTP request and returns a single response. The clients are unaware of each other — each one receives its response through its own socket connection as if it had been served individually.

### Deployment Topology

```
                        Internet
                            |
                            v
                +---------------------+
                |        Nginx        |
                |  TLS termination    |
                |  Rate limiting      |
                |  Load balancing     |
                +---------------------+
                      |         |
               (upstream 1) (upstream 2)
                      |
                      v
            +--------------------+
            |   req-batching     |   <-- this proxy
            |   reverse-proxy    |
            |   :8080            |
            +--------------------+
                |           |
         +------+           +------+
         |                         |
         v                         v
  +-------------+          +-------------+
  |    Redis    |          |    Kafka    |
  |  Dedup cache|          |  Telemetry  |
  |  TTL store  |          |  Replay buf |
  +-------------+          +-------------+
                      |
                      v
            +--------------------+
            |   Upstream Server  |
            |  (any HTTP service)|
            +--------------------+
```

### Nginx as Front-End

Nginx handles everything that should happen before a request reaches the batching layer:

- **TLS termination** — Nginx decrypts HTTPS and forwards plain HTTP to the proxy over a local socket or loopback. The batching proxy only ever sees HTTP/1 traffic.
- **Rate limiting** — `limit_req_zone` in Nginx caps requests per IP before they enter the batching queue, preventing a single client from monopolizing batch slots.
- **Load balancing** — Multiple instances of the batching proxy can run behind an Nginx `upstream` block. Nginx distributes connections across instances using round-robin or least-connections.

Example Nginx configuration for fronting the proxy:

```nginx
upstream batch_proxy {
    least_conn;
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
}

server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate     /etc/ssl/certs/example.crt;
    ssl_certificate_key /etc/ssl/private/example.key;

    location / {
        proxy_pass         http://batch_proxy;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 5s;
    }
}
```

The proxy is transparent to Nginx — it looks like any other HTTP upstream. No special Nginx modules or patches are required.

### Redis — Dedup Cache and In-Flight State

Redis integrates at the `serve_batch()` layer to short-circuit redundant upstream calls:

- **Response dedup** — Before forwarding a batch to the upstream, `serve_batch()` checks a Redis key constructed from the `BatchKey` fingerprint (method + path + optional request hash). If a cached response exists and has not expired (TTL-gated), the upstream call is skipped entirely and the cached body is fanned out to all waiting clients.
- **In-flight state** — A Redis key is written atomically at the start of a batch dispatch and deleted on completion. If a second instance of the proxy is processing an identical batch concurrently (in a multi-node deployment), it can detect the in-flight key and park its senders to await the first instance's result rather than issuing a duplicate upstream call.
- **TTL-based expiry** — Cache entries expire automatically. No explicit invalidation logic is needed for read-heavy, idempotent endpoints.

### Kafka — Telemetry and Replay Buffer

Every batch event is published to a Kafka topic for observability and RL training data:

- **Batch events** — On each flush, a structured record is emitted containing: `batch_key`, `batch_size`, `batch_age_ms`, `trigger` (timer or size cap or RL agent), `upstream_latency_ms`, and `timestamp`.
- **RL training data** — The Gymnasium environment consumes these Kafka records as experience tuples `(state, action, reward)`. The state vector is `[queue_depth, arrival_rate, batch_age_ms, upstream_p99_ms]`. The reward signal penalizes latency and rewards throughput.
- **Replay buffer** — Kafka's retention window acts as a durable experience replay buffer. The RL agent can re-train on historical traffic distributions without requiring a live traffic replay.
- **Audit trail** — Every batching decision is recorded, making it possible to reconstruct the exact sequence of requests and responses for any time window.

### Placing the Proxy Before Any Upstream

The batching proxy makes no assumptions about the upstream server. It forwards HTTP requests and propagates HTTP responses. This means it works in front of:

- REST APIs (any language or framework)
- GraphQL servers
- gRPC-transcoded HTTP/1 endpoints
- Static file servers
- Internal microservices

The only requirement is that the upstream speaks HTTP/1.1 over TCP. The upstream receives one request per batch — identical to what it would receive without the proxy — and the proxy fans the single response back to all N waiting clients.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for the full text.
