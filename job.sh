#!/bin/bash

#SBATCH -A m844
#SBATCH -t 00:2:00
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH --qos=debug

#SBATCH -o qout.256.%j # std::out 输出到这个文件 
#SBATCH -e qout.256.%j 

#SBATCH --mail-type=end,fail
#SBATCH --mail-user=youjia@northwestern.edu

if test "x$SLURM_NTASKS_PER_NODE" = x ; then #number of cores per node
   SLURM_NTASKS_PER_NODE=64 # 256 maximum
fi

NUM_NODES=$SLURM_JOB_NUM_NODES

NP=$((NUM_NODES * SLURM_NTASKS_PER_NODE))

ulimit -c unlimited
PRELOAD="/pscratch/sd/y/yll6162/soft/romio-install/lib/libromio.so:/usr/lib64/liblustreapi.so"
EXE=/global/homes/y/yll6162/E3SM-IO/src/e3sm_io 
SB_EXE=/tmp/${USER}_test #tmp all users shared
sbcast -v ${EXE} ${SB_EXE} #executable copy to this position

# LD_PRELOAD=${PRELOAD} srun -n $NP -c $((256/$SLURM_NTASKS_PER_NODE)) --cpu_bind=cores ${SB_EXE} -o /pscratch/sd/y/yll6162/FS_1M_4/can_F_out.nc /pscratch/sd/y/yll6162/FS_1M_4/map_f_case_16p.nc
LD_PRELOAD=${PRELOAD} srun -n $NP -c $((256/$SLURM_NTASKS_PER_NODE)) --cpu_bind=cores ${SB_EXE} -o /pscratch/sd/y/yll6162/FS_1M_64/can_F_out.nc /global/homes/y/yll6162/E3SM_dataset/map_f_case_21600p.nc
# 8 process per node => 256/8=32
