#!/usr/bin/env bash
# Run in the venv to we have the coverage tool.
source venv/bin/activate

pytest --cov=. "$@"
# Generate HTML report.
coverage html