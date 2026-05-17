/**
 * Test 4: RL Agent vs Fixed Timer Comparison
 *
 * Purpose: The core hypothesis — prove the PPO agent adapts to traffic
 * shape better than a naive fixed-timer. This test runs two traffic
 * phases that stress each policy differently:
 *
 *   Phase A — Sparse traffic (5 VUs):
 *     Fixed timer always waits the full 50ms even for lone requests.
 *     RL agent should flush early → lower p50.
 *
 *   Phase B — Burst traffic (200 VUs):
 *     Fixed timer flushes at 50ms regardless of batch fullness.
 *     RL agent should hold longer under high arrival rate → higher
 *     upstream reduction.
 *
 * NOTE: This test measures the LIVE RL agent behaviour. To compare
 * against fixed timer, run the proxy twice:
 *   Run 1: docker-compose up (RL agent active)       → save results/04_rl_agent.json
 *   Run 2: docker-compose up with RL_ENABLED=false   → save results/04_fixed_timer.json
 * Then diff the two result files.
 *
 * Pass criteria (RL agent run):
 *   Sparse phase p50 < 50ms  (agent flushes early, beats the timer)
 *   Burst phase p99 <= 65ms  (agent holds but never exceeds hard limit)
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";
import exec from "k6/execution";

// Separate trends for sparse vs burst phases
const sparsLatency = new Trend("sparse_latency_ms", true);
const burstLatency = new Trend("burst_latency_ms", true);
const errorRate = new Rate("error_rate");

export const options = {
  scenarios: {
    // Phase A: sparse — simulates low-traffic endpoint
    sparse_phase: {
      executor: "constant-vus",
      vus: 5,
      duration: "30s",
      startTime: "0s",
      tags: { phase: "sparse" },
    },
    // Phase B: burst — simulates bursty concurrent traffic
    burst_phase: {
      executor: "ramping-vus",
      startTime: "40s", // 10s gap after sparse phase ends
      stages: [
        { duration: "5s",  target: 200 },
        { duration: "30s", target: 200 },
        { duration: "5s",  target: 0   },
      ],
      tags: { phase: "burst" },
    },
  },
  thresholds: {
    "sparse_latency_ms{phase:sparse}": ["p(50)<50"],
    "burst_latency_ms{phase:burst}":   ["p(99)<=65"],
    error_rate: ["rate<0.01"],
  },
};

const PROXY_URL = "http://localhost:8080/api/resource";

export default function (data) {
  const phase = exec.scenario.name;
  const res = http.get(PROXY_URL);

  if (phase === "sparse_phase") {
    sparsLatency.add(res.timings.duration);
    sleep(1); // sparse = one request per second per VU
  } else {
    burstLatency.add(res.timings.duration);
    sleep(0.05);
  }

  errorRate.add(res.status !== 200);
  check(res, { "status 200": (r) => r.status === 200 });
}

export function handleSummary(data) {
  const sparseP50 =
    data.metrics["sparse_latency_ms"]?.values["p(50)"]?.toFixed(2) ?? "N/A";
  const sparseP99 =
    data.metrics["sparse_latency_ms"]?.values["p(99)"]?.toFixed(2) ?? "N/A";
  const burstP50 =
    data.metrics["burst_latency_ms"]?.values["p(50)"]?.toFixed(2) ?? "N/A";
  const burstP99 =
    data.metrics["burst_latency_ms"]?.values["p(99)"]?.toFixed(2) ?? "N/A";
  const errors = (data.metrics.error_rate?.values.rate * 100).toFixed(2);

  console.log("\n===== RL AGENT vs FIXED TIMER TEST ======");
  console.log("SPARSE PHASE (5 VUs — agent should flush early):");
  console.log(`  p50: ${sparseP50}ms  (target: <50ms)`);
  console.log(`  p99: ${sparseP99}ms`);
  console.log("");
  console.log("BURST PHASE (200 VUs — agent should batch aggressively):");
  console.log(`  p50: ${burstP50}ms`);
  console.log(`  p99: ${burstP99}ms  (target: <=65ms)`);
  console.log("");
  console.log(`error rate: ${errors}%`);
  console.log("-----------------------------------------");
  console.log("To compare against fixed timer baseline:");
  console.log("  1. Set RL_ENABLED=false in docker-compose.yml");
  console.log("  2. Re-run: k6 run 04_rl_vs_fixed_timer.js");
  console.log("  3. Diff results/04_rl_agent.json vs results/04_fixed_timer.json");
  console.log("=========================================\n");

  return {
    "results/04_rl_vs_fixed_timer.json": JSON.stringify(data, null, 2),
  };
}
