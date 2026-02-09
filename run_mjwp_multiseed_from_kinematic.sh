#!/usr/bin/env bash
set -euo pipefail

# Run MJWP retargeting multiple times with different seeds, starting from
# already-generated kinematic retargeting data (trajectory_kinematic*.npz).
#
# Each run writes into a fresh per-seed subfolder so artifacts don't overwrite.
# By default: output_dir = <processed_dir_robot>/seed_<seed>
#
# Usage examples:
#   ./run_mjwp_multiseed_from_kinematic.sh \
#     --run-dir /home/ROCQ/willow/fspinola/Documents/spider \
#     --dataset-dir /home/fspinola/example_datasets_fede \
#     --dataset-name gigahand \
#     --embodiment-type bimanual \
#     --robots allegro,inspire,ability,xhand,schunk \
#     --seeds 0,1,2,3,4
#
# Notes:
# - Requires the processed structure created by the SPIDER preprocess pipeline:
#   <dataset_dir>/processed/<dataset_name>/<robot>/<embodiment>/<task>/<data_id>/trajectory_kinematic*.npz

RUN_DIR=""
DATASET_DIR=""
DATASET_NAME=""
EMBODIMENT_TYPE="bimanual"
ROBOTS_CSV=""
SEEDS_CSV="0,1,2,3,4"
OVERRIDE=""
CONTACT_GUIDANCE=0

usage() {
  echo "Usage: $0 --run-dir <path> --dataset-dir <path> --dataset-name <name> [--embodiment-type <type>] [--robots r1,r2] [--seeds s0,s1,..] [--override <hydra_override>] [--contact-guidance]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2;;
    --dataset-dir) DATASET_DIR="$2"; shift 2;;
    --dataset-name) DATASET_NAME="$2"; shift 2;;
    --embodiment-type) EMBODIMENT_TYPE="$2"; shift 2;;
    --robots) ROBOTS_CSV="$2"; shift 2;;
    --seeds) SEEDS_CSV="$2"; shift 2;;
    --override) OVERRIDE="$2"; shift 2;;
    --contact-guidance) CONTACT_GUIDANCE=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$RUN_DIR" || -z "$DATASET_DIR" || -z "$DATASET_NAME" ]]; then
  usage
  exit 2
fi

IFS=',' read -r -a ROBOTS <<< "${ROBOTS_CSV}"
if [[ ${#ROBOTS[@]} -eq 0 || -z "${ROBOTS[0]:-}" ]]; then
  echo "--robots is required (comma-separated list)" >&2
  exit 2
fi

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
if [[ ${#SEEDS[@]} -eq 0 || -z "${SEEDS[0]:-}" ]]; then
  echo "--seeds must be a comma-separated list" >&2
  exit 2
fi

PROCESSED_ROOT="${DATASET_DIR%/}/processed/${DATASET_NAME}"

# Discover trials from the first robot (tasks/data_ids should match across robots).
DISCOVERY_ROBOT="${ROBOTS[0]}"
DISCOVERY_ROOT="${PROCESSED_ROOT}/${DISCOVERY_ROBOT}/${EMBODIMENT_TYPE}"

if [[ ! -d "$DISCOVERY_ROOT" ]]; then
  echo "Processed directory not found: $DISCOVERY_ROOT" >&2
  echo "Did you run preprocess (generate_xml + ik) already?" >&2
  exit 1
fi

if [[ $CONTACT_GUIDANCE -eq 1 ]]; then
  KIN_FILE="trajectory_kinematic_act.npz"
else
  KIN_FILE="trajectory_kinematic.npz"
fi

if [[ -z "$OVERRIDE" ]]; then
  OVERRIDE="$DATASET_NAME"
fi

cd "$RUN_DIR"

# Iterate over all <task>/<data_id> that have the kinematic file.
shopt -s nullglob
TASK_DIRS=("$DISCOVERY_ROOT"/*)
shopt -u nullglob

for task_dir in "${TASK_DIRS[@]}"; do
  [[ -d "$task_dir" ]] || continue
  task="$(basename "$task_dir")"

  shopt -s nullglob
  DATA_ID_DIRS=("$task_dir"/*)
  shopt -u nullglob

  for data_id_dir in "${DATA_ID_DIRS[@]}"; do
    [[ -d "$data_id_dir" ]] || continue
    data_id="$(basename "$data_id_dir")"

    # Only run if the kinematic NPZ exists for this trial.
    if [[ ! -f "$data_id_dir/$KIN_FILE" ]]; then
      continue
    fi

    echo "Trial found: task=$task data_id=$data_id (has $KIN_FILE)"

    for robot in "${ROBOTS[@]}"; do
      base_dir="${PROCESSED_ROOT}/${robot}/${EMBODIMENT_TYPE}/${task}/${data_id}"

      if [[ ! -d "$base_dir" ]]; then
        echo "  [skip] missing processed dir for robot=$robot: $base_dir" >&2
        continue
      fi
      if [[ ! -f "$base_dir/$KIN_FILE" ]]; then
        echo "  [skip] missing kinematic file for robot=$robot: $base_dir/$KIN_FILE" >&2
        continue
      fi

      for seed in "${SEEDS[@]}"; do
        # Requirement: seed and task ID are the same.
        # Here we encode that by using the seed both as RNG seed and as the per-run folder id.
        task_id="$seed"
        out_dir="${base_dir}/seed_${task_id}"
        mkdir -p "$out_dir"

        echo "  - robot=$robot seed=$seed output_dir=$out_dir"

        uv run --active examples/run_mjwp.py \
          +override="$OVERRIDE" \
          dataset_dir="$DATASET_DIR" \
          dataset_name="$DATASET_NAME" \
          embodiment_type="$EMBODIMENT_TYPE" \
          task="$task" \
          data_id="$data_id" \
          robot_type="$robot" \
          seed="$seed" \
          output_dir="$out_dir" \
          show_viewer=false \
          save_video=false
      done
    done
  done
done

echo "Done. Per-seed outputs are under: ${PROCESSED_ROOT}/<robot>/<embodiment>/<task>/<data_id>/seed_<seed>/"
