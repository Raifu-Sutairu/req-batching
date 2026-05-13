fn main() {
    tonic_build::compile_protos("proto/rl_agent.proto")
        .expect("Failed to compile protobufs");
}
