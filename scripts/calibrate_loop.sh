#!/bin/bash
# Re-run calibrate.py until every recording has been judged (judge results are cached, so retries are cheap).
# Useful when the LLM API is flaky / rate limited. Usage: scripts/calibrate_loop.sh [extra calibrate args]
cd "$(dirname "$0")/.." && . .venv/bin/activate
for i in $(seq 1 20); do
  python calibrate.py recordings --use-vtt --runs 1 "$@" > .cache/calib_final.log 2>&1
  if grep -q "confusion" .cache/calib_final.log; then echo "COMPLETE after $i attempts" >> .cache/calib_status.log; exit 0; fi
  echo "attempt $i incomplete $(date +%T)" >> .cache/calib_status.log; sleep 90
done
