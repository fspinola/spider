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
DATASET_NAME="gigahand"
EMBODIMENT_TYPE="bimanual"
robot_type=("allegro" "inspire" "ability" "xhand" "schunk") 
RAW_DATA_DIR="$DATASET_PATH/raw/$DATASET_NAME/hand_poses"

# activate the virtual environment
source /home/fspinola/venvs/spider-venv/bin/activate

cd "$RAW_DATA_DIR" || exit

for folder in p*; do
    # Extract fields using hyphens as separators
    participant=$(echo "$folder" | cut -d'-' -f1)    # p36
    scene=$(echo "$folder" | cut -d'-' -f2)          # tea
    seq=$(echo "$folder" | cut -d'-' -f3)            # 0010
    seq_int=$((10#$seq))                             # Convert to integer to remove leading zeros

    task="${participant}-${scene}"

    echo "Processing Folder: $folder"
    echo "  participant: $participant"
    echo "  scene:       $scene"
    echo "  sequence-id: $seq"
    echo "  task:        $task"
    echo

    (
        cd "$RUN_DIR" || exit
        # read data from dataset
        echo "  ----- Reading data from dataset"
        uv run --active python spider/process_datasets/gigahand.py \
                        --dataset-dir "$DATASET_PATH" \
                        --participant "$participant" \
                        --scene "$scene" \
                        --embodiment-type="$EMBODIMENT_TYPE" \
                        --sequence-id "$seq" --use-example-dataset

        # decompose object
        echo "  ----- Decomposing object"
        uv run --active python spider/preprocess/decompose_fast.py \
                        --dataset-dir "$DATASET_PATH" \
                        --dataset-name "$DATASET_NAME" \
                        --embodiment-type "$EMBODIMENT_TYPE" \
                        --task "$task" \
                        --data-id "$seq"

        # detect contact (optional)
        echo "  ----- Detecting contact"
        uv run --active python spider/preprocess/detect_contact.py \
                        --dataset-dir "$DATASET_PATH" \
                        --dataset-name "$DATASET_NAME" \
                        --embodiment-type "$EMBODIMENT_TYPE" \
                        --task "$task" \
                        --data-id "$seq" \
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
                            --data-id "$seq" \
                            --robot-type "$robot"

            # generate scene
            echo "    - Generating scene for contact guidance"
            uv run --active python spider/preprocess/generate_xml.py \
                            --dataset-dir "$DATASET_PATH" \
                            --dataset-name "$DATASET_NAME" \
                            --embodiment-type "$EMBODIMENT_TYPE" \
                            --task "$task" \
                            --data-id "$seq" \
                            --robot-type "$robot" \
                            --act_scene
            
            # kinematic retargeting
            echo "    - Kinematic retargeting"
            uv run --active spider/preprocess/ik.py \
                            --dataset-dir "$DATASET_PATH" \
                            --dataset-name "$DATASET_NAME" \
                            --embodiment-type "$EMBODIMENT_TYPE" \
                            --task "$task" \
                            --data-id "$seq" \
                            --robot-type "$robot" \
                            --save_video \
                            --open-hand
            
            # retargeting spider
            echo "    - Running retargeting spider"
            uv run --active examples/run_mjwp.py \
                            +override="$DATASET_NAME" \
                            dataset_dir="$DATASET_PATH" \
                            task="$task" \
                            data_id="$seq_int" \
                            robot_type="$robot" \
                            embodiment_type="$EMBODIMENT_TYPE" \
                            dataset_name="$DATASET_NAME"
            
            # # retargeting spider with contact guidance
            # echo "    - Running retargeting spider"
            # uv run --active examples/run_mjwp.py \
            #                 +override="${DATASET_NAME}_act" \
            #                 dataset_dir="$DATASET_PATH" \
            #                 task="$task" \
            #                 data_id="$seq_int" \
            #                 robot_type="$robot" \
            #                 embodiment_type="$EMBODIMENT_TYPE" \
            #                 dataset_name="$DATASET_NAME"
        done  
    )  
done    