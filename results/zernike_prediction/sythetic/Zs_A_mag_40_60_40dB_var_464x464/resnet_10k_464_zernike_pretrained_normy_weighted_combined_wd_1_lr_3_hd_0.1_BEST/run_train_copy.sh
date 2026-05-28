#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=05:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=128gb:gpu_mem=32gb:scratch_local=32gb:cl_bee=True
#PBS -N resnet_10k_464_zernike_pretrained_normy_weighted_combined_wd_1_lr_3_hd_0.1
#PBS -m abe
# initialize the required application (e.g. Python, version 3.4.1, compiled by gcc)
umask 0002
trap 'clean_scratch' TERM EXIT

TYPE=zernike_ftm

FILE=training_zernike_phase_pytorch.py 
CONFIG_FILE=train_config.json
DATADIR=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/training/Zs_A_mag_40_60_40dB_var_464x464//resnet_10k_464_zernike_pretrained_normy_weighted_combined_wd_1_lr_3_hd_0.1

DATADIR_INTENSITY=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A/output/intensity_circular.npz
DATADIR_PHASE=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A/output/phases_circular.npz
DATADIR_LABELS=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A/output/target_coefficients.npz
DATADIR_CARRIERS=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A/output/tilt.npz
DATADIR_FFT=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A/output/fft_log_norm.npz
DATADIR_FFT_NO_DC=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A/output/fft_log_norm_minus_dc.npz

MODELS=models.py
INTERFEROGRAM_GENERATION=interferogram.py

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

echo "Old HOME is $HOME"
export HOME=/storage/vestec1-elixir/home/$USER
echo "New HOME is $HOME"

cp $DATADIR/$MODELS $SCRATCHDIR
cp $DATADIR/$INTERFEROGRAM_GENERATION $SCRATCHDIR
cp $DATADIR/polynomials.py $SCRATCHDIR

cp $DATADIR/$FILE $SCRATCHDIR
cp $DATADIR/$CONFIG_FILE $SCRATCHDIR

#cd $DATADIR/output $SCRATCHDIR
cp $DATADIR_INTENSITY $SCRATCHDIR
cp $DATADIR_PHASE $SCRATCHDIR
cp $DATADIR_LABELS $SCRATCHDIR
cp $DATADIR_CARRIERS $SCRATCHDIR
#cp $DATADIR_FFT $SCRATCHDIR
#cp $DATADIR_FFT_NO_DC $SCRATCHDIR

cd $SCRATCHDIR

mkdir -p output
chmod  755 output

module add python/3.6.2-gcc

VENV_DIR=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/venv
source $VENV_DIR/bin/activate

#python -c "import torch; print(torch.__version__, torch.version.cuda)"

#export TMPDIR=${SCRATCHDIR}/pip-tmp
#export PIP_CACHE_DIR=${SCRATCHDIR}/pip-cache
#export XDG_CACHE_HOME=${SCRATCHDIR}/pip-cache
#mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
#python -m pip install --no-cache-dir --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

#echo "=== Ověření instalace ==="
#python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python3 $FILE

#cp -r $SCRATCHDIR/output $DATADIR

if [ -d "$SCRATCHDIR/output" ]; then
    echo "Output directory exists. Copying files..."
    cp -r $SCRATCHDIR/output $DATADIR
else
    echo "Output directory does not exist."
fi

clean_scratch
exit


