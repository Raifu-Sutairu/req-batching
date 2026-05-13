#!/bin/bash
# demo_high_traffic.sh
echo "=== HIGH TRAFFIC DEMO: Watch batching kick in ==="
echo "Sending 50 concurrent GET requests to the same endpoint..."

for i in $(seq 1 50); do
  curl -s http://localhost:8080/api/resource -o /dev/null &
done
wait

echo ""
echo "Check logs: you should see batch_size > 1 and flush_reason=rl_agent or timeout"
echo "Check Grafana at http://localhost:3000 — batch_size_at_flush histogram shows large batches"