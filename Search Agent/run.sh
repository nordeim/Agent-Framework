#!/usr/bin/env bash
set -euo pipefail

# Load environment
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs -d '\n')
fi

python brave_mcp_agent.py
