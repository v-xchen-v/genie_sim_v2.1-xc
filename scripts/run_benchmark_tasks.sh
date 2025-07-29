#!/bin/bash

TASKS=(
    "iros_clear_the_countertop_waste"
    "iros_restock_supermarket_items"
    "iros_clear_table_in_the_restaurant"
    "iros_stamp_the_seal"
    "iros_pack_moving_objects_from_conveyor"
    "iros_make_a_sandwich"
    # "iros_pack_in_the_supermarket"
)

CONTAINER_NAME="genie_sim_benchmark_v2.1"
SIM_COMMAND="omni_python server/source/genie.sim.lab/raise_standalone_sim.py"
BENCHMARK_COMMAND_BASE="omni_python ./benchmark/task_benchmark.py"
SOURCE_ENV="source ~/.bashrc"

for TASK_NAME in "${TASKS[@]}"; do
    echo "=== Running task: $TASK_NAME ==="

    # Run simulator in background inside container
    docker exec -d $CONTAINER_NAME bash -ic "$SOURCE_ENV && $SIM_COMMAND"
    echo ">>> Launched simulator in background"
    sleep 5  # let it settle

    # Run benchmark in background
    docker exec $CONTAINER_NAME bash -ic "$SOURCE_ENV && $BENCHMARK_COMMAND_BASE --task_name $TASK_NAME" &
    BENCH_PID=$!

    echo ">>> Waiting for sim or benchmark to finish..."

    while true; do
        # Benchmark done?
        if ! kill -0 $BENCH_PID 2>/dev/null; then
            echo ">>> Benchmark finished"
            break
        fi

        # Sim still alive in container?
        docker exec $CONTAINER_NAME pgrep -f raise_standalone_sim.py > /dev/null
        if [ $? -ne 0 ]; then
            echo ">>> Simulator exited"
            break
        fi

        sleep 2
    done

    echo ">>> Cleaning up"
    docker exec $CONTAINER_NAME pkill -f -9 raise_standalone_sim.py
    docker exec $CONTAINER_NAME pkill -f -9 task_benchmark.py

    sleep 2
done

echo "=== All tasks completed ==="
