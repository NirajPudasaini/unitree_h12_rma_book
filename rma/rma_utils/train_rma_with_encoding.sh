#!/bin/bash
# Script to train RMA policy with encoder and decoder

# Default parameters
TASK="Unitree-H12-Walk-RMA-v0"
NUM_ENVS=4096
MAX_ITERATIONS=5000
ENCODER_LOSS_WEIGHT=0.01
DECODER_LOSS_WEIGHT=0.1
SAVE_ENCODER_EVERY=100
SAVE_DECODER_EVERY=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --encoder-loss-weight)
            ENCODER_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --decoder-loss-weight)
            DECODER_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --save-encoder-every)
            SAVE_ENCODER_EVERY="$2"
            shift 2
            ;;
        --save-decoder-every)
            SAVE_DECODER_EVERY="$2"
            shift 2
            ;;
        --video)
            VIDEO="--video"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run training script
python train_rma_with_encoding.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --encoder_loss_weight "$ENCODER_LOSS_WEIGHT" \
    --decoder_loss_weight "$DECODER_LOSS_WEIGHT" \
    --save_encoder_every "$SAVE_ENCODER_EVERY" \
    --save_decoder_every "$SAVE_DECODER_EVERY" \
    $VIDEO
