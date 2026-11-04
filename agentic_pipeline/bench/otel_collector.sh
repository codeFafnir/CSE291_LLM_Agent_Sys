#!/usr/bin/env bash
# Start (or reuse) the OTLP collector that receives vLLM request spans on 127.0.0.1:4317.
# Spans append to /data/cse291-results/otel/spans.jsonl; run_csv.py slices them by run window.
set -euo pipefail

NAME=cse291-otel
IMAGE=otel/opentelemetry-collector-contrib:0.116.1
OUT_DIR=/data/cse291-results/otel
CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/otel_collector.yaml"

mkdir -p "${OUT_DIR}"

if [[ "$(docker inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null || true)" == "true" ]]; then
  exit 0
fi
docker rm -f "${NAME}" > /dev/null 2>&1 || true

docker run -d --name "${NAME}" --network host --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -v "${OUT_DIR}:/out" -v "${CONFIG}:/etc/otelcol-contrib/config.yaml:ro" \
  "${IMAGE}" --config /etc/otelcol-contrib/config.yaml > /dev/null

for _ in $(seq 1 30); do
  if (exec 3<> /dev/tcp/127.0.0.1/4317) 2>/dev/null; then exit 0; fi
  sleep 1
done
echo "otel collector ${NAME} did not open 127.0.0.1:4317" >&2
docker logs "${NAME}" >&2
exit 1
