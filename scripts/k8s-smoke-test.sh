#!/usr/bin/env bash
# End-to-end test on a Kubernetes cluster (kind in CI), using the fake backend:
#   1. deploy the kind overlay and wait for every rollout
#   2. upload audio through the API and wait for the result
#   3. upload again, force-delete the worker pod mid-job, and check the job
#      is recovered by another worker and still completes
set -euo pipefail
NS=meeting-analyzer
kubectl apply -k k8s/overlays/kind
for d in deploy/api deploy/worker deploy/ui statefulset/redis; do
  kubectl -n "$NS" rollout status "$d" --timeout=180s
done

kubectl -n "$NS" port-forward svc/api 8000:8000 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 30); do curl -sf localhost:8000/readyz >/dev/null && break; sleep 1; done

ffmpeg -loglevel error -y -f lavfi -i sine=d=2 /tmp/a.wav
ffmpeg -loglevel error -y -f lavfi -i sine=f=880:d=2 /tmp/b.wav

field() { python3 -c "import json,sys; print(json.load(sys.stdin)$1)"; }
wait_for() {  # wait_for <job id> <status> <seconds>
  for _ in $(seq "$3"); do
    s=$(curl -sf "localhost:8000/v1/meetings/$1" | field "['status']")
    [ "$s" = "$2" ] && return 0
    sleep 1
  done
  echo "job $1 did not reach $2 (last: $s)"; curl -s "localhost:8000/v1/meetings/$1"; return 1
}

echo "--- happy path"
ID=$(curl -sf -F file=@/tmp/a.wav localhost:8000/v1/meetings | field "['id']")
wait_for "$ID" done 90
curl -sf "localhost:8000/v1/meetings/$ID" | field "['result']['action_items']"

echo "--- worker pod killed mid-job"
ID=$(curl -sf -F file=@/tmp/b.wav localhost:8000/v1/meetings | field "['id']")
wait_for "$ID" processing 30
# The job could be on either worker, so force-delete all worker pods.
kubectl -n "$NS" delete pod -l app=worker --grace-period=0 --force --wait=false
wait_for "$ID" done 150
ATTEMPTS=$(curl -sf "localhost:8000/v1/meetings/$ID" | field "['attempts']")
echo "recovered after $ATTEMPTS attempts"
[ "$ATTEMPTS" -ge 2 ]
echo "kubernetes smoke test passed"
