#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 8
#SBATCH -n 1024

# 运行前:
#   pip install shyaml
# 用法:
#   bash run_wrf_batch.sh [config.yaml]
#
# 说明:
#   1) 仅做一次 WPS + real.exe，先生成统一前处理文件:
#      namelist.input, wrfbdy_d01, wrfinput_d01, wrfinput_d02
#   2) 在同一个 exp_dir 下创建 t01..t32 子目录用于批次运行 wrf.exe
#   3) 默认每个 wrf 任务 32 核；默认并发数 4（适配 1 节点 128 核）

set -euo pipefail

CONFIG_FILE="${1:-config.yaml}"
DEBUG_MODE="${DEBUG_MODE:-0}"
if [[ "$DEBUG_MODE" == "1" ]]; then
  set -x
fi

log() {
  echo "[$(date +'%F %T')] $*"
}

#================ 从 yaml 读取配置 ================
namelist_wps=$(shyaml get-value namelist_wps < "$CONFIG_FILE")
namelist_input=$(shyaml get-value namelist_input < "$CONFIG_FILE")
fnl_dir=$(shyaml get-value fnl_dir < "$CONFIG_FILE")
WPS=$(shyaml get-value WPS < "$CONFIG_FILE")
wrf_root=$(shyaml get-value wrf_root < "$CONFIG_FILE")
experiment_root=$(shyaml get-value experiment_root < "$CONFIG_FILE")
RESTART_DIR=$(shyaml get-value REATART_DIR < "$CONFIG_FILE" 2>/dev/null || true)
ant_dir=$(shyaml get-value ant_dir < "$CONFIG_FILE" 2>/dev/null || true)
vprm_diri=$(shyaml get-value vprm_diri < "$CONFIG_FILE" 2>/dev/null || shyaml get-value vprm_dir < "$CONFIG_FILE" 2>/dev/null || true)

#================ 批次参数（可在 config.yaml 中配置） ================
TOTAL_CASES=$(shyaml get-value total_cases < "$CONFIG_FILE" 2>/dev/null || echo 32)
CORES_PER_RUN=$(shyaml get-value cores_per_run < "$CONFIG_FILE" 2>/dev/null || echo 32)
RUNS_PER_NODE=$(shyaml get-value runs_per_node < "$CONFIG_FILE" 2>/dev/null || echo 4)

# 并发数 = 节点数 * 每节点并发
SLURM_NNODES="${SLURM_NNODES:-1}"
MAX_CONCURRENT=$((SLURM_NNODES * RUNS_PER_NODE))

timestamp=$(date +"%Y%m%d_%H%M")
exp_dir="$experiment_root/$timestamp"
wps_dir="$exp_dir/WPS"

mkdir -p "$exp_dir"
log "创建实验目录: $exp_dir"
log "总案例数: $TOTAL_CASES, 每任务核数: $CORES_PER_RUN, 最大并发: $MAX_CONCURRENT"
log "关键路径: WPS=$WPS, wrf_root=$wrf_root, experiment_root=$experiment_root"
log "可选路径: RESTART_DIR=${RESTART_DIR:-<empty>}, ant_dir=${ant_dir:-<empty>}, vprm_diri=${vprm_diri:-<empty>}"

#================ 拷贝 WPS 并更新 namelist ================
cp -r "$WPS" "$wps_dir"
python3 "$experiment_root/update_namelist_wps.py" "$namelist_wps" "$wps_dir"

#================ 准备 FNL 文件 ================
fnl_selected_dir="$wps_dir/FNL_SELECTED"
mkdir -p "$fnl_selected_dir"
python3 "$experiment_root/select_fnl.py" "$namelist_wps" "$fnl_dir" "$fnl_selected_dir"

#================ 运行 WPS ================
cd "$wps_dir"
srun -n "$CORES_PER_RUN" ./geogrid.exe > geogrid.log 2>&1
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
srun -n 1 ./link_grib.csh "$fnl_selected_dir"/*
srun -n 1 ./ungrib.exe > ungrib.log 2>&1
srun -n 1 ./metgrid.exe > metgrid.log 2>&1

#================ 生成统一前处理场（real.exe） ================
cp "$wrf_root"/run/* "$wps_dir"
python3 "$experiment_root/update_namelist_input.py" "$namelist_input" "$wps_dir"

srun -n "$CORES_PER_RUN" ./real.exe > real.log 2>&1

for required_file in namelist.input wrfbdy_d01 wrfinput_d01 wrfinput_d02; do
  if [[ ! -f "$wps_dir/$required_file" ]]; then
    echo "ERROR: 缺少前处理文件 $required_file"
    exit 1
  fi
done

log "统一前处理文件已生成完成"

#================ 创建 t01..tNN 并分发文件 ================
for i in $(seq -w 1 "$TOTAL_CASES"); do
  case_dir="$exp_dir/t$i"
  mkdir -p "$case_dir"

  # 复制 WRF run 必需静态文件与可执行文件
  cp "$wrf_root"/run/* "$case_dir"/

  # 分发统一前处理文件（本次 namelist 不区分）
  cp "$wps_dir/namelist.input" "$case_dir/"
  cp "$wps_dir/wrfbdy_d01" "$case_dir/"
  cp "$wps_dir/wrfinput_d01" "$case_dir/"
  cp "$wps_dir/wrfinput_d02" "$case_dir/"
done

log "已创建 t01..t$(printf "%02d" "$TOTAL_CASES")"

# 在创建完 t01..tNN 后，再向每个子目录复制重启文件（如存在）
if [[ -n "${RESTART_DIR:-}" ]] && ls "$RESTART_DIR"/wrfrst* >/dev/null 2>&1; then
  for i in $(seq -w 1 "$TOTAL_CASES"); do
    case_dir="$exp_dir/t$i"
    cp "$RESTART_DIR"/wrfrst* "$case_dir"/
  done
  log "已向每个 t 目录复制 wrfrst* 重启文件:"
  ls -1 "$RESTART_DIR"/wrfrst* | xargs -n1 basename
else
  log "未检测到 wrfrst*，各 t 目录按冷启动文件运行"
fi

# 在创建完 t01..tNN 后，逐目录生成扰动人为源/vprm 文件（不再执行 real.exe）
if [[ -n "${ant_dir:-}" && ! -d "$ant_dir" ]]; then
  echo "ERROR: ant_dir 不存在: $ant_dir"
  exit 1
fi

if [[ -n "${vprm_diri:-}" && ! -d "$vprm_diri" ]]; then
  echo "ERROR: vprm_diri 不存在: $vprm_diri"
  exit 1
fi

for i in $(seq -w 1 "$TOTAL_CASES"); do
  case_dir="$exp_dir/t$i"
  case_id=$((10#$i))

  if [[ -n "${ant_dir:-}" ]]; then
    before_ant=$(find "$case_dir" -maxdepth 1 -type f -name 'wrfchemi_*' | wc -l)
    log "[t$i] 执行 ant_move.py (before wrfchemi count=$before_ant)"
    python3 "$experiment_root/ant_move.py" "$case_dir" "$case_id"
    after_ant=$(find "$case_dir" -maxdepth 1 -type f -name 'wrfchemi_*' | wc -l)
    log "[t$i] ant_move.py 完成 (after wrfchemi count=$after_ant)"
  fi

  if [[ -n "${vprm_diri:-}" ]]; then
    output_dir="$case_dir"
    before_vprm=$(find "$case_dir" -maxdepth 1 -type f -name 'vprm_input_*' | wc -l)
    log "[t$i] 执行 vprm_move.py (before vprm count=$before_vprm)"
    python3 "$experiment_root/vprm_move.py" "$output_dir" "$namelist_wps" "$vprm_diri"
    after_vprm=$(find "$case_dir" -maxdepth 1 -type f -name 'vprm_input_*' | wc -l)
    log "[t$i] vprm_move.py 完成 (after vprm count=$after_vprm)"
    if (( after_vprm == before_vprm )); then
      log "[t$i] WARNING: vprm 文件数量未变化，请检查 vprm_move.py 日志与源目录时间范围"
    fi
  fi
done

#================ 批次并发运行 wrf.exe ================
run_case() {
  local case_dir="$1"
  (
    cd "$case_dir"
    log "[$(basename "$case_dir")] 启动 wrf.exe"
    srun --exclusive -N 1 -n "$CORES_PER_RUN" ./wrf.exe > wrf.log 2>&1
    log "[$(basename "$case_dir")] wrf.exe 完成"
  )
}

wait_for_slot() {
  while true; do
    current_jobs=$(jobs -rp | wc -l)
    if (( current_jobs < MAX_CONCURRENT )); then
      break
    fi
    sleep 1
  done
}

pids=()
for i in $(seq -w 1 "$TOTAL_CASES"); do
  wait_for_slot
  run_case "$exp_dir/t$i" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "ERROR: 存在 wrf.exe 子任务失败，请检查各 tXX/wrf.log"
  exit 1
fi

echo ">>> 全部批次完成，结果位于: $exp_dir/t01..t$(printf "%02d" "$TOTAL_CASES")"
