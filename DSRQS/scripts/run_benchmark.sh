#!/bin/bash

echo "========================================"
echo "   DSRQS FULL BENCHMARK START"
echo "========================================"

# Prepare data splits
python main.py --dataset omim_hop3 --mode prepare_data
python main.py --dataset orphanet_fq274 --mode prepare_data
python main.py --dataset disgenet_rd411 --mode prepare_data

# Run experiments
for DATASET in omim_hop3 orphanet_fq274 disgenet_rd411
do
    echo "Running dataset: $DATASET"
    python main.py --dataset $DATASET --mode full_eval
done

echo "========================================"
echo "   BENCHMARK DONE"
echo "========================================"