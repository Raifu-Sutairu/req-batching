#!/bin/bash
mkdir -p proto_out
python3 -m grpc_tools.protoc -I../reverse-proxy/proto --python_out=./proto_out --grpc_python_out=./proto_out ../reverse-proxy/proto/rl_agent.proto
echo "Protobuf files generated successfully in rl/proto_out/"
