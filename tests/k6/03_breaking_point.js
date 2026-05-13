/**
 * Test 3: Breaking Point — Maximum Throughput
 *
 * Purpose: Find the maximum sustained request rate the proxy handles
 * before p99 latency degrades or error rate climbs. Demonstrates
 * production robustness of the Rust/Kafka/Redis/RL stack.
 *
 * Traffic shape:
 *   Stepped ramp — hold each level for 20s to let metrics stabilise.
 *   50 → 100 → 200 → 400 → 600 → 800 VUs
 *   Then ramp down to observe recovery.
 *
 * Pass criteria:
 *   Max throughput before p99 > 200ms or error rate > 2%
 *   (this threshold is intentionally lenient — we want to find the cliff)
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

const latency = new Trend("proxy_latency_ms", true);
const errorRate = new Rate("error_rate");

export const options = {
  stages: [
    { duration: "10s", target: 50 },
    { duration: "20s", target: 50 },
    { duration: "10s", target: 100 },
    { duration: "20s", target: 100 },
    { duration: "10s", target: 200 },
    { duration: "20s", target: 200 },
    { duration: "10s", target: 400 },
    { duration: "20s", target: 400 },
    { duration: "10s", target: 600 },
    { duration: "20s", target: 600 },
    { duration: "10s", target: 800 },
    { duration: "20s", target: 800 },
    { duration: "15s", target: 0 }, // recovery
  ],
  // No hard thresholds — we WANT to observe degradation
  thresholds: {},
};

const PROXY_URL = "http://localhost:8080/api/resource";

export default function () {
  const res = http.get(PROXY_URL);

  latency.add(res.timings.duration);
  errorRate.add(res.status !== 200);

  check(res, { "status 200": (r) => r.status === 200 });

  sleep(0.05);
}

export function handleSummary(data) {
  const p50 = data.metrics.proxy_latency_ms.values["p(50)"];
  const p90 = data.metrics.proxy_latency_ms.values["p(90)"];
  const p99 = data.metrics.proxy_latency_ms.values["p(99)"];
  const errors = data.metrics.error_rate.values.rate * 100;
  const peakRps = data.metrics.http_reqs.values.rate;
  const totalReqs = data.metrics.http_reqs.values.count;

  console.log("\n========== BREAKING POINT TEST =========");
  console.log(`total requests : ${totalReqs}`);
  console.log(`peak req/s     : ${peakRps.toFixed(1)}`);
  console.log(`p50 latency    : ${p50.toFixed(2)}ms`);
  console.log(`p90 latency    : ${p90.toFixed(2)}ms`);
  console.log(`p99 latency    : ${p99.toFixed(2)}ms`);
  console.log(`error rate     : ${errors.toFixed(2)}%`);
  console.log("-----------------------------------------");
  if (errors > 2) {
    console.log("⚠  ERROR RATE EXCEEDED 2% — breaking point reached");
  } else if (p99 > 200) {
    console.log("⚠  p99 EXCEEDED 200ms — latency cliff reached");
  } else {
    console.log("✓  Stack held under 800 VUs — no breaking point found");
  }
  console.log("=========================================\n");

  return {
    "results/03_breaking_point.json": JSON.stringify(data, null, 2),
  };
}
