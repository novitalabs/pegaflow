#!/bin/bash
set -e

mkdir -p /tmp/fio-ssd-0 /tmp/fio-ssd-1

run_fio() {
    local name=$1
    local rw=$2
    shift 2
    echo "--- Running $name ($rw) ---"
    fio \
        --name="bench" \
        --ioengine=sync \
        --direct=1 \
        --bs=4k \
        --size=64m \
        --numjobs=4 \
        --runtime=10 \
        --time_based \
        --rw="$rw" \
        --group_reporting \
        --output-format=json \
        "$@" \
        2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
jobs = data['jobs']
for j in jobs:
    d = j['$rw']
    bw = d['bw'] / 1024.0
    iops = d['iops']
    lat = d['lat_ns']['mean'] / 1000.0
    print(f'  Throughput: {bw:.1f} MiB/s')
    print(f'  IOPS:       {iops:.0f}')
    print(f'  Avg latency: {lat:.1f} us')
"
    find /tmp/fio-ssd-0 /tmp/fio-ssd-1 -name 'bench.*' -delete 2>/dev/null || true
}

echo "# SSD Multi-Path Throughput Benchmark"
echo ""
echo "Environment: overlay filesystem (same underlying device)"
echo "Workload: 4 KiB blocks, 64 MiB per job, 4 parallel jobs, 10 s runtime"
echo ""

# Write benchmarks
run_fio "1path_1shard" "write" --directory=/tmp/fio-ssd-0 --nrfiles=1
run_fio "1path_2shard" "write" --directory=/tmp/fio-ssd-0 --nrfiles=2
run_fio "2path_1shard" "write" --directory=/tmp/fio-ssd-0 --directory=/tmp/fio-ssd-1 --nrfiles=1
run_fio "2path_2shard" "write" --directory=/tmp/fio-ssd-0 --directory=/tmp/fio-ssd-1 --nrfiles=2

# Read benchmarks
run_fio "1path_1shard" "read" --directory=/tmp/fio-ssd-0 --nrfiles=1
run_fio "1path_2shard" "read" --directory=/tmp/fio-ssd-0 --nrfiles=2
run_fio "2path_1shard" "read" --directory=/tmp/fio-ssd-0 --directory=/tmp/fio-ssd-1 --nrfiles=1
run_fio "2path_2shard" "read" --directory=/tmp/fio-ssd-0 --directory=/tmp/fio-ssd-1 --nrfiles=2

rm -rf /tmp/fio-ssd-0 /tmp/fio-ssd-1
