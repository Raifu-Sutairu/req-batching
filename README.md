# Request Batch Dispatching using RL-Based Algorithms

A high-performance, protocol-agnostic reverse proxy in Rust that uses a **Reinforcement Learning agent** to make intelligent request batching decisions — deciding *when* to flush a batch based on latency/throughput trade-offs rather than fixed timers.

---

## Project Layout

```
req-batching/
├── reverse-proxy/      # Core proxy engine (Rust)
│   └── src/
│       ├── main.rs         # Entry point, wires config + state + listener
│       ├── config.rs       # Static configuration (addr, timeouts, limits)
│       ├── state.rs        # Shared AppState (DashMap batch registry)
│       ├── batch.rs        # BatchKey, BatchState, BatchSlot types
│       ├── router.rs       # Routing decision logic (Batch vs PassThrough)
│       ├── listener.rs     # TCP accept loop, connection limiting, graceful shutdown
│       └── service.rs      # Hyper HTTP handler, batching engine, serve_batch()
├── rl/                 # RL agent (planned)
└── docs/               # Design notes
```

---

## What's Done

### ✅ TCP Listener (`listener.rs`)
- Binds to a configurable `SocketAddr` and runs a non-blocking accept loop
- **Semaphore-based connection cap** — pauses acceptance when at `max_connections` to protect memory
- **Graceful shutdown** via `tokio::sync::watch` — drains in-flight requests before closing connections
- Per-connection `tokio::spawn` tasks with backoff on accept errors

### ✅ HTTP/1 Service Layer (`listener.rs` + `service.rs`)
- Each accepted TCP stream is handed to a `hyper` HTTP/1 server
- Requests are dispatched through a custom `handle_request` service function
- The connection future is `select!`-ed against the shutdown signal for clean teardown

### ✅ Router (`router.rs`)
- Classifies every incoming request into one of two decisions:
  - `RoutingDecision::Batch(BatchKey)` — for `GET` requests (grouped by method + path)
  - `RoutingDecision::PassThrough` — for all other methods
- Easily extensible — swap in path-prefix rules, header inspection, etc.

### ✅ Request Batching Engine (`service.rs`, `batch.rs`, `state.rs`)
- Concurrent batch registry: `DashMap<BatchKey, Arc<Mutex<BatchSlot>>>`
  - `DashMap` for shard-locked concurrent access across Tokio tasks
  - `std::sync::Mutex` (not Tokio's) for the inner slot — pure memory ops, no async needed
- Incoming requests park themselves by pushing a `oneshot::Sender` into the slot, then `await` the receiver
- **Two flush triggers**, whichever fires first:
  - ⏱ **Timer**: first request in a batch spawns a background task that sleeps for `batch_timeout_ms` then calls `serve_batch()`
  - 📦 **Size cap**: if `senders.len() >= max_batch_size`, `serve_batch()` is called immediately inline
- `serve_batch()` is idempotent via a `Waiting → Serving` state transition — safe against races between the timer task and the size-cap path
- On flush: removes the slot from the map (no new requests can join), drains all senders, and broadcasts the response

### ✅ Configuration (`config.rs`)
| Field | Description |
|---|---|
| `listen_addr` | Socket to bind (`127.0.0.1:8080` by default) |
| `max_connections` | Global connection cap (semaphore) |
| `batch_timeout_ms` | Max time to hold a batch open before flushing |
| `max_batch_size` | Max requests per batch before flushing early |

---

## What's Stubbed / Next

| Item | Status | Notes |
|---|---|---|
| Upstream HTTP forwarding | 🔲 Planned | All responses are currently mock strings. Real forwarding needs an HTTP client (e.g. `hyper-util` client or `reqwest`) to relay requests and pipe the upstream response back through the `oneshot` senders |
| Example upstream server | 🔲 Planned | A small example server to test against end-to-end |
| `created_at` on `BatchSlot` | 🔲 Unused | Stored but not yet read — will feed into RL state features (batch age) |
| Tracing initialisation | 🔲 Planned | `tracing::info!` calls exist; `tracing_subscriber` not yet wired in `main` |
| Config from file / env | 🔲 Planned | Currently hardcoded dummy values in `main` |
| **RL agent** | 🔲 Planned | Will replace the fixed timer/size triggers with a learned policy |
| gRPC support | 🔲 Planned | HTTP/2 + protobuf framing for gRPC upstreams |
| Kafka integration | 🔲 Planned | Batch → Kafka topic dispatch path |
| Redis integration | 🔲 Planned | Response caching / dedup layer |
| nginx-style config | 🔲 Planned | Declarative upstream routing rules |

---

## Dependencies

| Crate | Purpose |
|---|---|
| `tokio` | Async runtime (full features) |
| `hyper` | HTTP/1 server + request/response types |
| `hyper-util` | `TokioIo` adapter, future HTTP client |
| `dashmap` | Concurrent shard-locked hash map for batch registry |
| `tracing` | Structured logging |

---

## Running

```bash
cd reverse-proxy
cargo run
# Proxy listens on 127.0.0.1:8080
```

```bash
# Test a batched GET request
curl http://127.0.0.1:8080/some/path

# Test a passthrough (non-GET)
curl -X POST http://127.0.0.1:8080/some/path
```

---

## Design Goals

- **Protocol-agnostic core** — the batching engine is decoupled from the upstream transport. Adding gRPC, Kafka, or Redis dispatch is a matter of plugging in a new `serve_batch` variant
- **RL-driven flush policy** — replace hardcoded `batch_timeout_ms` / `max_batch_size` triggers with an agent that observes batch age, size, and upstream latency to learn an optimal flush policy
- **Zero-copy where possible** — `std::mem::take` drains sender vecs without cloning; `Arc<Mutex<>>` avoids duplicating slot data across tasks
