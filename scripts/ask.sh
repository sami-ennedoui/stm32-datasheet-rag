#!/usr/bin/env bash
# Send a question to the running API and pretty-print the JSON answer.
# Usage: scripts/ask.sh "How is the USART baud rate configured on STM32H7?"
set -euo pipefail
Q="${1:-How is the USART baud rate configured on STM32H7?}"
URL="${STM32_API_URL:-http://127.0.0.1:8000}"
curl -fsS -X POST "${URL}/ask" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$Q")" \
  | python3 -m json.tool
