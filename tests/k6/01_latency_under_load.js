/**
 * Test 1: p50 / p99 Latency Under Load
 *
 * Purpose: Prove that the batching proxy + gRPC RL agent overhead
 * does not introduce unacceptable latency under sustained load.
 *
 * Traffic shape:
 *   - Ramp from 0 → 50 VUs over 15s   (warm-up)
 *   - Hold at 50 VUs for 30s           (steady state)
 *   - Ramp from 50 → 150 VUs over 15s  (stress)
 *   - Hold at 150 VUs for 30s          (stress steady state)
 *   - Ramp down to 0 over 10s
 *
 * Pass criteria:
 *   p50 < 60ms, p99 < 120ms, error rate < 1%
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

const latency = new Trend("proxy_latency_ms", true);
const errorRate = new Rate("error_rate");

export const options = {
  stages: [
    { duration: "15s", target: 50 },
    { duration: "30s", target: 50 },
    { duration: "15s", target: 150 },
    { duration: "30s", target: 150 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    proxy_latency_ms: ["p(50)<60", "p(99)<120"],
    error_rate: ["rate<0.01"],
  },
};

const PROXY_URL = "http://localhost:8080/api/resource";

export default function () {
  const res = http.get(PROXY_URL);

  latency.add(res.timings.duration);
  errorRate.add(res.status !== 200);

  check(res, {
    "status 200": (r) => r.status === 200,
    "p99 under 120ms": (r) => r.timings.duration < 120,
  });

  sleep(0.1);
}

export function handleSummary(data) {
  const p50 = data.metrics.proxy_latency_ms.values["p(50)"];
  const p99 = data.metrics.proxy_latency_ms.values["p(99)"];
  const errors = data.metrics.error_rate.values.rate * 100;
  const rps = data.metrics.http_reqs.values.rate;

  console.log("\n========== LATENCY UNDER LOAD ==========");
  console.log(`p50 latency  : ${p50.toFixed(2)}ms`);
  console.log(`p99 latency  : ${p99.toFixed(2)}ms`);
  console.log(`error rate   : ${errors.toFixed(2)}%`);
  console.log(`throughput   : ${rps.toFixed(1)} req/s`);
  console.log("=========================================\n");

  return {
    "results/01_latency_under_load.json": JSON.stringify(data, null, 2),
  };
}
