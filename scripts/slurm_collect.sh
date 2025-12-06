#!/bin/bash
#SBATCH --job-name=fmkv_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err

# FMKVC Trajectory Collection - SLURM Job Script
# For AMD Milan 96-core nodes with 1TB memory
#
# Usage:
#   sbatch scripts/slurm_collect.sh
#
# To check progress:
#   tail -f logs/collect_<jobid>.out

echo "=================================================="
echo "FMKVC Trajectory Collection - SLURM Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Load required modules (adjust for your HPC)
# module load python/3.10
# module load cuda/12.0  # If using GPU

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4  # Limit OpenMP threads per worker
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false  # Avoid HF tokenizer warnings

# Create log directory
mkdir -p logs

# Configuration
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR="./data/trajectories_100gb"
NUM_SAMPLES=1000000  # ~1M samples for ~100GB
NUM_WORKERS=16       # 16 workers × 6 cores each = 96 cores

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Samples: $NUM_SAMPLES"
echo "  Workers: $NUM_WORKERS"
echo ""

# Run parallel collection
python scripts/collect_trajectories_parallel.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --num_workers $NUM_WORKERS \
    --batch_size 4 \
    --max_seq_length 2048 \
    --collect_forces

# Check exit status
EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Trajectories saved to $OUTPUT_DIR/merged"

    # Print disk usage
    echo ""
    echo "Disk usage:"
    du -sh "$OUTPUT_DIR"
    du -sh "$OUTPUT_DIR/merged"

    # Count files
    echo ""
    echo "Files collected:"
    ls -la "$OUTPUT_DIR/merged/" | head -20
else
    echo "FAILED: Check logs for errors"
fi

echo "=================================================="
