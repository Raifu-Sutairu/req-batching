#!/bin/bash

PROXY_URL="http://localhost:8080/api/resource"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   req-batching Live Demo${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# --- Phase 1: Low Traffic ---
echo -e "${YELLOW}[Phase 1] Low Traffic — 3 sparse requests${NC}"
for i in 1 2 3; do
  curl -s -o /dev/null -w "  → request $i: HTTP %{http_code} in %{time_total}s\n" "$PROXY_URL"
  sleep 1
done

echo ""
echo -e "${GREEN}Sleeping 3s...${NC}"
sleep 3

# --- Phase 2: High Traffic (100 reqs) ---
echo ""
echo -e "${YELLOW}[Phase 2] High Traffic — 100 concurrent requests${NC}"
for i in $(seq 1 100); do
  curl -s -o /dev/null "$PROXY_URL" &
done
wait
echo -e "  → 100 requests fired and completed"

echo ""
echo -e "${GREEN}Sleeping 5s...${NC}"
sleep 5

# --- Phase 3: High Traffic (200 reqs) ---
echo ""
echo -e "${YELLOW}[Phase 3] High Traffic — 200 concurrent requests${NC}"
for i in $(seq 1 200); do
  curl -s -o /dev/null "$PROXY_URL" &
done
wait
echo -e "  → 200 requests fired and completed"

echo ""
echo -e "${GREEN}Sleeping 6s...${NC}"
sleep 6

# --- Phase 4: Low Traffic ---
echo ""
echo -e "${YELLOW}[Phase 4] Low Traffic — 3 sparse requests${NC}"
for i in 1 2 3; do
  curl -s -o /dev/null -w "  → request $i: HTTP %{http_code} in %{time_total}s\n" "$PROXY_URL"
  sleep 1
done

echo ""
echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   Demo complete.${NC}"
echo -e "${BLUE}   Check Grafana → http://localhost:3000${NC}"
echo -e "${BLUE}   Check logs   → docker-compose logs -f reverse-proxy${NC}"
echo -e "${BLUE}==========================================${NC}"