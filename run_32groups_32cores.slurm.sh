#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 8
#SBATCH -n 1024
# 运行前:  pip install shyaml  (用于解析yaml)

CONFIG_FILE="config.yaml"
set -e
#================ 从yaml读取配置 ================
namelist_wps=$(shyaml get-value namelist_wps < $CONFIG_FILE)
namelist_input=$(shyaml get-value namelist_input < $CONFIG_FILE)
fnl_dir=$(shyaml get-value fnl_dir < $CONFIG_FILE)
WPS=$(shyaml get-value WPS < $CONFIG_FILE)
wrf_root=$(shyaml get-value wrf_root < $CONFIG_FILE)
experiment_root=$(shyaml get-value experiment_root < $CONFIG_FILE)
MOZBC_DIR=$(shyaml get-value mozbc_dir < $CONFIG_FILE)
MOZBC_FILE_DIR=$(shyaml get-value MOZBC_FILE_DIR < $CONFIG_FILE)
RESTART_DIR=$(shyaml get-value REATART_DIR < $CONFIG_FILE)
ant_dir=$(shyaml get-value ant_dir < $CONFIG_FILE)
vprm_diri=$(shyaml get-value vprm_diri < $CONFIG_FILE)

run_one_case () {
    gid=$1

    #================ 创建新的实验目录 ================
    timestamp=$(date +"%m%d%H%M")
    exp_dir=$experiment_root/${timestamp}_g${gid}
    mkdir -p $exp_dir
    echo ">>> [group${gid}] 创建实验目录: $exp_dir"

    #================    拷贝 WPS    ================
    cp -r $WPS $exp_dir/WPS
    out_namelist_file=$exp_dir/WPS

    python3 $experiment_root/update_namelist_wps.py $namelist_wps $out_namelist_file

    #================ 准备 FNL 文件 ================
    output_dir=$exp_dir/WPS/FNL_SELECTED
    mkdir -p $output_dir

    python3 $experiment_root/select_fnl.py $namelist_wps $fnl_dir $output_dir

    #================ 运行 WPS =====================
    cd $exp_dir/WPS
    srun --exclusive -n 32 ./geogrid.exe > geogrid.log 2>&1
    ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
    srun --exclusive -n 1 ./link_grib.csh $output_dir/*
    srun --exclusive -n 1 ./ungrib.exe > ungrib.log 2>&1
    srun --exclusive -n 1 ./metgrid.exe > metgrid.log 2>&1

    #================    拷贝 WRF    ================
    cp  $wrf_root/run/* $exp_dir/WPS

    out_namelist_input_file=$exp_dir/WPS

    python3 $experiment_root/update_namelist_input.py $namelist_wps $out_namelist_input_file

    # 判断目标文件夹下是否存在wrfrst开头的文件
    if ls "$RESTART_DIR"/wrfrst* 1> /dev/null 2>&1; then
        # 存在则复制所有wrfrst开头的文件到当前路径
        cp "$RESTART_DIR"/wrfrst* ./
        echo " 已成功复制以下wrfrst文件到当前路径："
        ls -l "$RESTART_DIR"/wrfrst* | awk '{print $9}' | xargs -n1 basename
    else
        # 不存在则输出提示信息
        echo " 目标文件夹 $RESTART_DIR 下未找到wrfrst开头的文件"
    fi

    #================ 运行 WRF ===================
    srun --exclusive -n 32 ./real.exe > real.log 2>&1

    output_dir=$exp_dir/WPS

    python3 $experiment_root/ant_move.py $output_dir $namelist_wps $ant_dir

    python3 $experiment_root/vprm_move.py $output_dir $namelist_wps $vprm_diri

    #================ 添加初始场  =====================
    cd $MOZBC_DIR
    MOZBC_file=$MOZBC_DIR/MOZCART.inp
    for i in {1..2}; do
        python3 $experiment_root/update_mozbc.py $MOZBC_FILE_DIR $MOZBC_file $output_dir $i
        srun --exclusive -n 1 ./mozbc <MOZCART.inp
    done
    #================ 运行 WRF =====================

    cd $exp_dir/WPS
    srun --exclusive -n 32 ./wrf.exe

    echo ">>> [group${gid}] 实验完成，结果在 $exp_dir"
}

for gid in $(seq 1 32); do
    run_one_case $gid &
done
wait

echo ">>> 全部 32 组实验完成"
