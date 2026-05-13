FROM rust:latest as builder
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y protobuf-compiler cmake pkg-config zlib1g-dev

COPY . .
RUN cargo build --release
RUN cargo build --release --example echo_server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y openssl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/local/bin
COPY --from=builder /usr/src/app/target/release/reverse-proxy .
COPY --from=builder /usr/src/app/target/release/examples/echo_server .
COPY config.toml /usr/local/bin/

ENV PROXY_LISTEN_ADDR=0.0.0.0:8080
EXPOSE 8080 9090

CMD ["./reverse-proxy"]
