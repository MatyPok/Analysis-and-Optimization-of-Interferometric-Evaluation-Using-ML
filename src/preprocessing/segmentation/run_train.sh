#!/bin/bash
#PBS -q <SERVER>
#PBS -l walltime=<WALLTIME>
#PBS -l <RESOURCES>
#PBS -N <JOB_NAME>
#PBS -m abe
# initialize the required application (e.g. Python, version 3.4.1, compiled by gcc)
umask 0002
clean_scratch() {

    echo "Cleaning scratch directory..."

    if [ -n "$SCRATCHDIR" ] && [ -d "$SCRATCHDIR" ]; then
        rm -rf "$SCRATCHDIR"/*
    fi

}

trap 'cleanup_lock; clean_scratch' TERM EXIT

TYPE=<TYPE>

FILE=<FILE_NAME> 
SETUP=<SETUP>
CONFIG_FILE=<CONFIG_FILE>
PROJECT_DIR=<PROJECT_DIR>
TRAINING_DIR=<TRAINING_DIR>

DATADIR=<DATADIR>

DATADIR_TRAIN_IMAGES=<DATADIR_TRAIN_IMAGES>
DATADIR_TRAIN_GT_MASKS=<DATADIR_TRAIN_GT_MASKS>

DATADIR_VAL_IMAGES=<DATADIR_VAL_IMAGES>
DATADIR_VAL_GT_MASKS=<DATADIR_VAL_GT_MASKS>

if [ $TYPE == "refiner" ]; then
    DATADIR_TRAIN_INTERP_MASKS=<DATADIR_TRAIN_INTERP_MASKS>
    DATADIR_VAL_INTERP_MASKS=<DATADIR_VAL_INTERP_MASKS>
fi

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $TRAINING_DIR/jobs_info.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

echo "Old HOME is $HOME"
export HOME=/storage/vestec1-elixir/home/$USER
echo "New HOME is $HOME"

cp $TRAINING_DIR/$FILE $SCRATCHDIR
cp $TRAINING_DIR/$CONFIG_FILE $SCRATCHDIR

cp $PROJECT_DIR/$SETUP $SCRATCHDIR


#cp -r $DATADIR $SCRATCHDIR

cp -r $DATADIR_TRAIN_IMAGES $SCRATCHDIR
cp -r $DATADIR_TRAIN_GT_MASKS $SCRATCHDIR

cp -r $DATADIR_VAL_IMAGES $SCRATCHDIR
cp -r $DATADIR_VAL_GT_MASKS $SCRATCHDIR

if [ $TYPE == "refiner" ]; then
    cp -r $DATADIR_TRAIN_INTERP_MASKS $SCRATCHDIR
    cp -r $DATADIR_VAL_INTERP_MASKS $SCRATCHDIR
fi



cd $SCRATCHDIR

#echo $USER
mkdir -p $TRAINING_DIR/output
#sudo chown $USER:$USER $TRAINING_DIR/output
chmod 755 $TRAINING_DIR/output


module purge
module load python/3.10.4-gcc-8.3.0-ovkjwzd

PYTHON=$(which python)

VENV_DIR=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/venv
LOCKDIR="$VENV_DIR.lock"

cleanup_lock() {
    if [ -d "$LOCKDIR" ]; then
        rmdir "$LOCKDIR"
    fi
}


if [ ! -f "$VENV_DIR/bin/python" ]; then

    if mkdir "$LOCKDIR" 2>/dev/null; then

        echo "Creating shared virtualenv at $VENV_DIR"

        $PYTHON -m venv $VENV_DIR

        source $VENV_DIR/bin/activate

        export TMPDIR=$SCRATCHDIR/tmp
        export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

        mkdir -p $TMPDIR
        mkdir -p $PIP_CACHE_DIR

        python -m pip install --upgrade pip setuptools wheel

        $VENV_DIR/bin/python $SETUP

    else

        echo "Waiting for other job to finish venv creation..."

        while [ -d "$LOCKDIR" ]; do
            sleep 10
        done

    fi

fi

source $VENV_DIR/bin/activate

export TMPDIR=$SCRATCHDIR/tmp
export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache
export TORCH_HOME=$SCRATCHDIR/torch_cache
export HF_HOME=$SCRATCHDIR/hf_cache
export XDG_CACHE_HOME=$SCRATCHDIR/.cache

mkdir -p $TMPDIR
mkdir -p $PIP_CACHE_DIR
mkdir -p $TORCH_HOME
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

echo "Verifying environment..."

if ! $VENV_DIR/bin/python - <<EOF
import torch
print(torch.__version__)
assert torch.__version__.startswith("2.1")
EOF

then

    echo "Environment incomplete. Running setup..."

    export TMPDIR=$SCRATCHDIR/tmp
    export PIP_CACHE_DIR=$SCRATCHDIR/pip_cache

    mkdir -p $TMPDIR
    mkdir -p $PIP_CACHE_DIR

    python $PROJECT_DIR/$SETUP

fi

$VENV_DIR/bin/python -m pip list | grep segmentation

echo "Actual python:"
$VENV_DIR/bin/python -c "import sys; print(sys.executable)"

echo "Torch version:"
$VENV_DIR/bin/python -c "import torch; print(torch.__version__)"

echo "Segmentation models check:"
$VENV_DIR/bin/python -c "import segmentation_models_pytorch as smp; print(smp.__version__)"

#echo "Detectron2 check:"
#$VENV_DIR/bin/python -c "import detectron2; print(detectron2.__version__)"

# run training
$VENV_DIR/bin/python $SCRATCHDIR/$FILE


if [ -d "$TRAINING_DIR/output" ]; then
    echo "Output directory exists. Copying files..."
    cp -r $SCRATCHDIR/output/* $TRAINING_DIR/output/
else
    echo "Output directory does not exist. Creating and copying files..."
    mkdir -p $TRAINING_DIR/output
    cp -r $SCRATCHDIR/output/* $TRAINING_DIR/output/
fi

clean_scratch
exit


