#!/bin/bash
#PBS -N cnn_test
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=2:ngpus=1:mem=10gb:scratch_local=8gb
#PBS -l walltime=2:00:00
# The 4 lines above are options for scheduling system: job will run 2 hour at maximum, 1 machine with 2 processors + 10gb RAM memory + 8gb scratch memory are requested

set -e # stop the script if any command fails

trap 'clean_scratch' TERM EXIT

HOME_DIR=/storage/praha1/home/krejcji2
PROJECT_DIR=$HOME_DIR/metacentrum
INPUT_DATA_DIR=/data/ptbxl

# Check if the variable is defined and not empty
if [[ -z "${EXP_NAME}" ]]; then
    echo "Error: EXP_NAME is not defined." >&2  # Send error to stderr
    exit 1  # Exit with an error code
fi

# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

mkdir -p $SCRATCHDIR/experiments
cp -r $PROJECT_DIR/experiments/$EXP_NAME $SCRATCHDIR/experiments/$EXP_NAME || { echo >&2 "Error while copying input file(s)!"; exit 2; }
mkdir -p $SCRATCHDIR$INPUT_DATA_DIR
cp -rT $PROJECT_DIR$INPUT_DATA_DIR/ $SCRATCHDIR$INPUT_DATA_DIR/ || { echo >&2 "Error while copying input file(s)!"; exit 2; }
# cp -r $HOME_DIR/metacentrum/data/ptbxl/ scratch/data/ptbxl/
cp -r $PROJECT_DIR/src $SCRATCHDIR/src || { echo >&2 "Error while copying input file(s)!"; exit 2; }

mkdir -p $SCRATCHDIR/experiments/$EXP_NAME/outputs

# Export path of the user packages for the singularity container
export PYTHONUSERBASE=$HOME_DIR/.local2311
export SINGULARITY_CACHEDIR=$HOME_DIR
export LOCALCACHEDIR=$SCRATCHDIR

# Without GPU
#singularity exec '/cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.11-py3.SIF' python /src/optimize_optuna.py \
#        --model
#        --config /config/config.yaml \
#        --data_dir /data \
#        --results_dir /results \

cd $SCRATCHDIR
singularity exec --nv -H /storage/praha1/home/krejcji2 /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.11-py3.SIF python src/optimize_net.py $EXP_NAME

# move the output to user's DATADIR or exit in case of failure
cp -r $SCRATCHDIR/experiments/$EXP_NAME/outputs/ $PROJECT_DIR/experiments/$EXP_NAME/outputs/ || { echo >&2 "Error while copying output file(s)!"; exit 3; }
