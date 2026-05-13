import grpc
from concurrent import futures
import sys
import os
import time

# Add the generated files to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../proto_out'))

import rl_agent_pb2
import rl_agent_pb2_grpc

class RlAgentServicer(rl_agent_pb2_grpc.RlAgentServicer):
    def Decide(self, request, context):
        print("\n--- Received BatchState from Proxy ---")
        print(f"Batch Size:      {request.batch_size}")
        print(f"Batch Age (ms):  {request.batch_age_ms:.2f}")
        print(f"Upstream p99:    {request.upstream_p99_ms:.2f}")
        print(f"Request Rate:    {request.request_rate:.2f}/sec")
        print("--------------------------------------")
        
        # Dummy logic: always tell it to flush for testing
        response = rl_agent_pb2.FlushDecision()
        response.should_flush = True
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rl_agent_pb2_grpc.add_RlAgentServicer_to_server(RlAgentServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Python RL Agent Stub listening on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
