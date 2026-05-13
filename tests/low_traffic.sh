#!/bin/bash
# demo_low_traffic.sh
echo "=== LOW TRAFFIC DEMO: Watch agent flush immediately ==="

for i in 1 2 3; do
  echo "Request $i (2 second gap):"
  curl -s -w "  response time: %{time_total}s\n" http://localhost:8080/api/resource
  sleep 2
done

echo ""
echo "Check logs: batch_size=1, flush_reason=rl_agent (agent saw sparse traffic and flushed early)"