#!/bin/bash

"""
Script to process the GigaHands example dataset for all participants, scenes, and sequences, following the SPIDER pipeline.
For each data folder, it performs the following steps:
1. Reads data from the dataset.
2. Decomposes the object.
3. Detects contact (optional).
4. For each robot type, it generates the scene, performs kinematic retargeting, and runs the retargeting spider.    
"""

RUN_DIR="/home/ROCQ/willow/fspinola/Documents/spider"
DATASET_PATH="/home/fspinola/example_datasets_fede"
DATASET_NAME="oakink"
EMBODIMENT_TYPE="bimanual"
robot_type=("allegro") 
RAW_DATA_DIR="$DATASET_PATH/raw/$DATASET_NAME"

# activate the virtual environment
source /home/fspinola/venvs/spider-venv/bin/activate

cd "$RAW_DATA_DIR" || exit

for file in *.pkl; do
    # Extract fields using hyphens as separators
    task="${file%%_${EMBODIMENT_TYPE}*}"    # lift_board

    echo "Processing File: $file"
    echo "  task:        $task"
    echo

    (
        cd "$RUN_DIR" || exit
        # read data from dataset
        echo "  ----- Reading data from dataset"
        uv run --active python spider/process_datasets/oakink.py \
                        --dataset-dir="$DATASET_PATH" \
                        --embodiment-type="$EMBODIMENT_TYPE" \
                        --task="$task" \
                        --show_viewer \
                        --save_video

        # decompose object
        echo "  ----- Decomposing object"
        uv run --active python spider/preprocess/decompose_fast.py \
                        --dataset-dir "$DATASET_PATH" \
                        --dataset-name "$DATASET_NAME" \
                        --embodiment-type "$EMBODIMENT_TYPE" \
                        --task "$task" \
                        --data-id 0

        # detect contact (optional)
        echo "  ----- Detecting contact"
        uv run --active python spider/preprocess/detect_contact.py \
                        --dataset-dir "$DATASET_PATH" \
                        --dataset-name "$DATASET_NAME" \
                        --embodiment-type "$EMBODIMENT_TYPE" \
                        --task "$task" \
                        --data-id 0 \
                        --save-video 

        # Process below for each robot type
        for robot in "${robot_type[@]}"; do
            echo "  ----- Processing robot: $robot"
            # generate scene
            echo "    - Generating scene"
            uv run --active python spider/preprocess/generate_xml.py \
                            --dataset-dir "$DATASET_PATH" \
                            --dataset-name "$DATASET_NAME" \
                            --embodiment-type "$EMBODIMENT_TYPE" \
                            --task "$task" \
                            --data-id 0 \
                            --robot-type "$robot"

            # generate scene
            echo "    - Generating scene for contact guidance"
            uv run --active python spider/preprocess/generate_xml.py \
                            --dataset-dir "$DATASET_PATH" \
                            --dataset-name "$DATASET_NAME" \
                            --embodiment-type "$EMBODIMENT_TYPE" \
                            --task "$task" \
                            --data-id 0 \
                            --robot-type "$robot" \
                            --act_scene
            
            # kinematic retargeting
            echo "    - Kinematic retargeting"
            uv run --active spider/preprocess/ik.py \
                            --dataset-dir "$DATASET_PATH" \
                            --dataset-name "$DATASET_NAME" \
                            --embodiment-type "$EMBODIMENT_TYPE" \
                            --task "$task" \
                            --data-id 0 \
                            --robot-type "$robot" \
                            --save_video \
                            --open-hand
            
            # retargeting spider
            echo "    - Running retargeting spider"
            uv run --active examples/run_mjwp.py \
                            +override="${DATASET_NAME}_fast" \
                            dataset_dir="$DATASET_PATH" \
                            task="$task" \
                            data_id=0 \
                            robot_type="$robot" \
                            embodiment_type="$EMBODIMENT_TYPE" \
                            dataset_name="$DATASET_NAME"
            
            # # retargeting spider with contact guidance
            # echo "    - Running retargeting spider"
            # uv run --active examples/run_mjwp.py \
            #                 +override="${DATASET_NAME}_act" \
            #                 dataset_dir="$DATASET_PATH" \
            #                 task="$task" \
            #                 data_id=0 \
            #                 robot_type="$robot" \
            #                 embodiment_type="$EMBODIMENT_TYPE" \
            #                 dataset_name="$DATASET_NAME"
        done  
    )  
done    