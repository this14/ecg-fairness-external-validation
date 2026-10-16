#!/usr/bin/env bash
# run_seed_sweep.sh — BSPC 리비전용 다중 시드 재학습
#
# 리뷰어 2: "Conduct repeated training runs with different random seeds to
#            establish the stability of the reported performance differences."
#
# 격자: mode 2 (vanilla / debiased) x lead_set 3 (12lead / 3lead / 1lead)
#       x seed 10 = 60회
#
# 체크포인트는 시드별 폴더에 들어갑니다. 한 폴더에 쓰면 덮어써서
# 마지막 시드만 남고, 그러면 이 실험 전체가 무의미해집니다.
#
#   $OUT_ROOT/seed42/model_bbb_leadcount/...
#   $OUT_ROOT/seed43/model_bbb_leadcount/...
#
# 사용법
#     export DATA_DIR=/path/to/preprocessed      # X.npy, y.npy, groups.npy
#     export SPLIT_DIR=/path/to/split            # *_idx.npy
#     bash run_seed_sweep.sh
#
#     DRY=1 bash run_seed_sweep.sh               # 명령만 출력
#     SEEDS="42 43" bash run_seed_sweep.sh       # 일부만
#
# tmux 안에서 돌리십시오. 60회는 몇 시간입니다.
set -u

PY=${PY:-python}
SCRIPT=${SCRIPT:-train_bbb_leadcount_seeded.py}
DATA_DIR=${DATA_DIR:?DATA_DIR 를 지정하십시오 (X.npy 가 있는 폴더)}
SPLIT_DIR=${SPLIT_DIR:?SPLIT_DIR 를 지정하십시오 (*_idx.npy 가 있는 폴더)}
OUT_ROOT=${OUT_ROOT:-./seed_sweep}
EPOCHS=${EPOCHS:-15}
BATCH=${BATCH:-256}

MODES=(vanilla debiased)
LEADSETS=(12lead 3lead 1lead)
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49 50 51"}

# 논문 §2.4: 1-lead 최종 실행만 lr 5e-4, 나머지는 1e-3.
# 시드 비교가 성립하려면 이 설정이 조건마다 고정돼야 합니다.
lr_for () {
  if [ "$1" = "1lead" ]; then echo "5e-4"; else echo "1e-3"; fi
}

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/sweep_console.log"
nseed=$(echo $SEEDS | wc -w)
total=$(( ${#MODES[@]} * ${#LEADSETS[@]} * nseed ))
i=0; ran=0; skipped=0; failed=0
t0=$(date +%s)

{
echo "시드 스윕 $total 회 시작: $(date)"
echo "mode ${MODES[*]} | lead_set ${LEADSETS[*]} | seeds $SEEDS"
echo "data=$DATA_DIR  split=$SPLIT_DIR  out=$OUT_ROOT"
echo "----------------------------------------------------------------"
} | tee -a "$LOG"

for S in $SEEDS; do
  for M in "${MODES[@]}"; do
    for L in "${LEADSETS[@]}"; do
      i=$((i+1))
      OUT="$OUT_ROOT/seed${S}/${M}_${L}"
      LR=$(lr_for "$L")

      # 이미 체크포인트가 있으면 건너뛴다 (중단 후 재개용).
      # 조용한 skip 이 아니라 눈에 보이게 출력합니다.
      if ls "$OUT"/*.pt >/dev/null 2>&1; then
        echo "[$i/$total] seed$S $M $L -> 이미 있음, 건너뜀" | tee -a "$LOG"
        skipped=$((skipped+1)); continue
      fi

      mkdir -p "$OUT"
      CMD="$PY $SCRIPT \
        --data_dir $DATA_DIR --split_dir $SPLIT_DIR \
        --lead_set $L --mode $M --seed $S \
        --epochs $EPOCHS --batch_size $BATCH --lr $LR \
        --out_dir $OUT"

      if [ "${DRY:-0}" = "1" ]; then
        echo "[$i/$total] $CMD"
        continue
      fi

      echo "[$i/$total] seed$S $M $L (lr=$LR)" | tee -a "$LOG"
      if ! $CMD > "$OUT/train_console.log" 2>&1; then
        echo "  *** 실패: $OUT/train_console.log" | tee -a "$LOG"
        tail -5 "$OUT/train_console.log" | sed 's/^/    /' | tee -a "$LOG"
        failed=$((failed+1)); continue
      fi
      ran=$((ran+1))
      el=$(( $(date +%s) - t0 ))
      if [ "$ran" -gt 0 ]; then eta=$(( el / ran * (total - i) / 60 )); else eta=0; fi
      echo "  완료 (경과 $((el/60))분, 남은 약 ${eta}분)" | tee -a "$LOG"
    done
  done
done

{
echo "----------------------------------------------------------------"
echo "완료: $(date)   총 $(( ($(date +%s) - t0) / 60 ))분"
echo "새로 학습 $ran | 건너뜀 $skipped | 실패 $failed"
echo ""
echo "다음 단계"
echo "  1. 내부 평가:  시드별 폴더에 대해 evaluate.py"
echo "  2. 외부 검증:  EV_MODEL_DIR 를 시드별 폴더로 바꿔가며"
echo "                 run_external_validation.py 를 반복"
echo "     SNUH 50,000건 x 60모델이므로 GPU 로도 수 시간입니다."
echo "     EchoNext(4,626건) 먼저 돌려 방향을 확인한 뒤 SNUH 를 거십시오."
} | tee -a "$LOG"
