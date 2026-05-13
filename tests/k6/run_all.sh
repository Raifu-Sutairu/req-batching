#!/bin/bash
# =============================================================================
# k6 Test Suite Runner — req-batching
# Runs all 4 tests in sequence and prints a combined summary.
#
# Prerequisites:
#   - k6 installed: https://k6.io/docs/get-started/installation/
#   - docker-compose stack running: docker-compose up -d
#   - Wait ~10s after docker-compose up before running this
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Results saved to: tests/k6/results/
# =============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

cd "$SCRIPT_DIR" || exit 1
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}"
echo "=============================================="
echo "   req-batching k6 Test Suite"
echo "=============================================="
echo -e "${NC}"

# Health check before running
echo -e "${YELLOW}Checking proxy is up...${NC}"
if ! curl -sf http://localhost:8080/api/resource -o /dev/null; then
  echo -e "${RED}ERROR: Proxy not reachable at http://localhost:8080${NC}"
  echo "Run: docker-compose up -d && sleep 10"
  exit 1
fi
echo -e "${GREEN}✓ Proxy is up${NC}\n"

# Check k6 is installed
if ! command -v k6 &>/dev/null; then
  echo -e "${RED}ERROR: k6 not found.${NC}"
  echo "Install: https://k6.io/docs/get-started/installation/"
  exit 1
fi

run_test() {
  local num="$1"
  local name="$2"
  local file="$3"

  echo -e "${YELLOW}[$num/4] Running: $name${NC}"
  echo "----------------------------------------------"

  if k6 run \
    --out json="$RESULTS_DIR/$(basename "$file" .js)_raw.json" \
    "$SCRIPT_DIR/$file"; then
    echo -e "${GREEN}✓ $name passed${NC}\n"
  else
    echo -e "${RED}✗ $name — thresholds not met (see output above)${NC}\n"
  fi
}

run_test "1" "Latency Under Load"           "01_latency_under_load.js"
run_test "2" "Upstream Call Reduction"      "02_upstream_reduction.js"
run_test "3" "Breaking Point / Max RPS"     "03_breaking_point.js"
run_test "4" "RL Agent vs Fixed Timer"      "04_rl_vs_fixed_timer.js"

echo -e "${BLUE}"
echo "=============================================="
echo "   All tests complete."
echo "   Results saved to: tests/k6/results/"
echo ""
echo "   Grafana dashboard: http://localhost:3000"
echo "   Prometheus:        http://localhost:9091"
echo "=============================================="
echo -e "${NC}"