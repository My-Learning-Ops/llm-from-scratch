#!/bin/bash

# clear_checkpoints.sh
# A simple script to clear the contents of the checkpoints directory.
# Brendan Dileo, July 2025

CHECKPOINT_DIR="../checkpoints"

# Check if directory exists and clear its contents
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Clearing checkpoints in $CHECKPOINT_DIR..."
    rm -rf "$CHECKPOINT_DIR"/*
else
    echo "Checkpoint directory $CHECKPOINT_DIR does not exist."
fi