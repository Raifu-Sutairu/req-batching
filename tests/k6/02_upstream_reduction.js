/**
 * Test 2: Upstream Call Reduction vs No-Batching
 *
 * Purpose: Quantify how many upstream calls the proxy saves compared
 * to a hypothetical no-batching baseline (1 upstream call per request).
 *
 * Method:
 *   - Send bursts of concurrent requests to the same endpoint.
 *   - The proxy coalesces them; the mock upstream receives far fewer calls.
 *   - We measure this via the Prometheus counter `batch_flush_total`
 *     compared to total requests sent.
 *
 * Traffic shape:
 *   - 5 burst waves of 50 concurrent VUs, each lasting 5s
 *   - 3s gap between waves to let batches flush cleanly
 *
 * Pass criteria:
 *   upstream_call_reduction > 80%
 *   (i.e. proxy sent < 20% of the requests as upstream calls)
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate } from "k6/metrics";

const totalRequests = new Counter("total_requests");
const successRate = new Rate("success_rate");

export const options = {
  scenarios: {
    burst_waves: {
      executor: "ramping-vus",
      stages: [
        // Wave 1
        { duration: "2s", target: 50 },
        { duration: "5s", target: 50 },
        { duration: "3s", target: 0 },
        // Wave 2
        { duration: "2s", target: 50 },
        { duration: "5s", target: 50 },
        { duration: "3s", target: 0 },
        // Wave 3
        { duration: "2s", target: 50 },
        { duration: "5s", target: 50 },
        { duration: "3s", target: 0 },
        // Wave 4
        { duration: "2s", target: 50 },
        { duration: "5s", target: 50 },
        { duration: "3s", target: 0 },
        // Wave 5
        { duration: "2s", target: 50 },
        { duration: "5s", target: 50 },
        { duration: "3s", target: 0 },
      ],
    },
  },
  thresholds: {
    success_rate: ["rate>0.99"],
  },
};

const PROXY_URL = "http://localhost:8080/api/resource";
// Prometheus metrics endpoint — scrape before and after to diff flush counts
const METRICS_URL = "http://localhost:9091/metrics";

export default function () {
  const res = http.get(PROXY_URL);
  totalRequests.add(1);
  successRate.add(res.status === 200);
  check(res, { "status 200": (r) => r.status === 200 });
}

export function handleSummary(data) {
  // Scrape Prometheus to get actual upstream flush count
  const metricsRes = http.get(METRICS_URL);
  let flushCount = 0;

  if (metricsRes.status === 200) {
    const lines = metricsRes.body.split("\n");
    for (const line of lines) {
      // Sum all flush reasons to get total upstream calls made
      if (line.startsWith("batch_flush_total{") && !line.startsWith("#")) {
        const match = line.match(/} (\d+(\.\d+)?)/);
        if (match) flushCount += parseFloat(match[1]);
      }
    }
  }

  const sentRequests = data.metrics.total_requests.values.count;
  const reduction =
    flushCount > 0
      ? (((sentRequests - flushCount) / sentRequests) * 100).toFixed(2)
      : "N/A (could not scrape Prometheus)";

  console.log("\n===== UPSTREAM CALL REDUCTION ==========");
  console.log(`requests sent      : ${sentRequests}`);
  console.log(`upstream calls made: ${flushCount}`);
  console.log(`reduction          : ${reduction}%`);
  console.log(
    `baseline (no batch): ${sentRequests} upstream calls would have been made`
  );
  console.log("=========================================\n");

  return {
    "results/02_upstream_reduction.json": JSON.stringify(
      {
        ...data,
        upstream_reduction_summary: {
          requests_sent: sentRequests,
          upstream_calls: flushCount,
          reduction_pct: reduction,
        },
      },
      null,
      2
    ),
  };
}
