#!/bin/bash
#PBS -l select=1:ncpus=2:mem=128gb:scratch_local=32gb
#PBS -l walltime=2:00:00
#PBS -N intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A
#PBS -m abe
# initialize the required application (e.g. Python, version 3.4.1, compiled by gcc)

trap 'clean_scratch' TERM EXIT
#DATADIR=/storage/projects/CVUT_Fsv_AO.old/ML_Interferometry/DATA/data_with_phases_227x227_35k
DATADIR=/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A
#DATADIR=/storage/projects/CVUT_Fsv_AO.old/ML_Interferometry/DATA/data_with_phases_227x227_20k_lam1_Z2Z
#DATADIR_confing = /storage/projects/CVUT_Fsv_AO/test_new_data_2025/data_gen_0_na_1/config.json
#DATADIR_data=/storage/projects/CVUT_Fsv_AO/NCK2_DP002_SHARE_AOG_UPOL/data_2025_CTU_test/intens20250110150143.npz
#DATADIR_labels=/storage/projects/CVUT_Fsv_AO/NCK2_DP002_SHARE_AOG_UPOL/data_2025_CTU_test/params20250110150143.npz

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp $DATADIR/interferogram_zernike_set_for_ftm.py $SCRATCHDIR
cp $DATADIR/config_0.json $SCRATCHDIR
cp $DATADIR/polynomials.py $SCRATCHDIR
cp $DATADIR/coefficients.py $SCRATCHDIR
cp $DATADIR/noise.py $SCRATCHDIR
#cp $DATADIR_data $SCRATCHDIR
#cp $DATADIR_labels $SCRATCHDIR
cd $SCRATCHDIR


#echo $USER
mkdir output
#sudo chown $USER:$USER output
chmod 755 output

module add python/3.6.2-gcc
#module add python/3.10.4-gcc-8.3.0-ovkjwzd

python3 interferogram_zernike_set_for_ftm.py

cp -r $SCRATCHDIR/output $DATADIR
clean_scratch
exit