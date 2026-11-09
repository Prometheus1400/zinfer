#!/bin/sh
# Start a disposable local Prometheus + Grafana session.
# Usage: ./observability/run.sh
#        ZINFER_PORT=8081 ./observability/run.sh
# Ctrl+C stops both child processes and deletes this session's data and logs.
set -eu

observability_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for tool in prometheus grafana curl; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing $tool. Install the monitoring tools with: brew install prometheus grafana" >&2
        exit 1
    fi
done

ZINFER_PORT=${ZINFER_PORT:-8080}
ZINFER_PROMETHEUS_PORT=${ZINFER_PROMETHEUS_PORT:-9090}
ZINFER_GRAFANA_PORT=${ZINFER_GRAFANA_PORT:-3000}
for port in "$ZINFER_PORT" "$ZINFER_PROMETHEUS_PORT" "$ZINFER_GRAFANA_PORT"; do
    case "$port" in
        ''|*[!0-9]*) echo "Invalid port: $port" >&2; exit 1 ;;
    esac
    if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        echo "Port must be between 1 and 65535: $port" >&2
        exit 1
    fi
done

# Homebrew's Grafana binary needs the directory containing conf/defaults.ini
# and the frontend assets. GRAFANA_HOMEPATH also supports standalone installs.
grafana_home=${GRAFANA_HOMEPATH:-}
if [ -z "$grafana_home" ] && command -v brew >/dev/null 2>&1; then
    grafana_prefix=$(brew --prefix grafana)
    grafana_home="$grafana_prefix/share/grafana"
fi
if [ ! -f "$grafana_home/conf/defaults.ini" ]; then
    echo "Set GRAFANA_HOMEPATH to the directory containing Grafana's conf/defaults.ini." >&2
    exit 1
fi

session_dir=$(mktemp -d "${TMPDIR:-/tmp}/zinfer-observability.XXXXXX")
prometheus_pid=
grafana_pid=
cleanup() {
    exit_code=$?
    trap - EXIT INT TERM HUP
    for pid in "$prometheus_pid" "$grafana_pid"; do
        if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
    done
    for pid in "$prometheus_pid" "$grafana_pid"; do
        if [ -n "$pid" ]; then wait "$pid" 2>/dev/null || true; fi
    done
    if [ "$exit_code" -ne 0 ] && [ "$exit_code" -ne 130 ] && [ "$exit_code" -ne 143 ]; then
        for log in "$session_dir/plugin-install.log" "$session_dir/prometheus.log" "$session_dir/grafana.log"; do
            if [ -f "$log" ]; then
                echo "Log: $log" >&2
                tail -n 20 "$log" >&2
            fi
        done
    fi
    rm -rf -- "$session_dir"
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

export ZINFER_GRAFANA_PORT ZINFER_PROMETHEUS_PORT
export ZINFER_OBSERVABILITY_DIR="$observability_dir"
export ZINFER_OBSERVABILITY_SESSION="$session_dir"
cp -R "$observability_dir/provisioning" "$session_dir/provisioning"
mkdir -p "$session_dir/provisioning/alerting" "$session_dir/provisioning/plugins"
sed "s/127.0.0.1:8080/127.0.0.1:$ZINFER_PORT/" \
    "$observability_dir/prometheus.yml" > "$session_dir/prometheus.yml"

# Grafana 13.2 packages Prometheus as a standalone plugin. Pin it and install
# it before provisioning the datasource; its files are disposable too.
echo "Preparing the Prometheus datasource plugin..."
grafana cli --homepath "$grafana_home" --config "$observability_dir/grafana.ini" \
    --pluginsDir "$session_dir/plugins" \
    plugins install prometheus 13.2.5 > "$session_dir/plugin-install.log" 2>&1

prometheus \
    --config.file="$session_dir/prometheus.yml" \
    --storage.tsdb.path="$session_dir/prometheus" \
    --storage.tsdb.retention.time=2h \
    --web.listen-address="127.0.0.1:$ZINFER_PROMETHEUS_PORT" \
    --log.level=warn > "$session_dir/prometheus.log" 2>&1 &
prometheus_pid=$!

grafana server \
    --homepath "$grafana_home" \
    --config "$observability_dir/grafana.ini" \
    > "$session_dir/grafana.log" 2>&1 &
grafana_pid=$!

# Wait for both tools, reporting startup failures through the cleanup logs.
attempt=0
while :; do
    if ! kill -0 "$prometheus_pid" 2>/dev/null || ! kill -0 "$grafana_pid" 2>/dev/null; then
        echo "A monitoring process exited during startup. Check for occupied ports." >&2
        exit 1
    fi
    if curl -fsS --max-time 1 "http://127.0.0.1:$ZINFER_PROMETHEUS_PORT/-/ready" >/dev/null 2>&1 &&
        curl -fsS --max-time 1 "http://127.0.0.1:$ZINFER_GRAFANA_PORT/api/health" >/dev/null 2>&1; then
        break
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "Timed out waiting for Prometheus and Grafana." >&2
        exit 1
    fi
    sleep 1
done

echo "Dashboard:  http://localhost:$ZINFER_GRAFANA_PORT/d/zinfer-engine/zinfer-engine"
echo "Prometheus: http://localhost:$ZINFER_PROMETHEUS_PORT"
echo "Scraping:   http://127.0.0.1:$ZINFER_PORT/metrics (every second)"
echo "Session:    $session_dir"
echo "Ctrl+C stops monitoring and deletes this session's history and logs."

while kill -0 "$prometheus_pid" 2>/dev/null && kill -0 "$grafana_pid" 2>/dev/null; do
    sleep 1
done
echo "A monitoring process exited unexpectedly." >&2
exit 1
