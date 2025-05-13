#!/usr/bin/env bash
set -euo pipefail

ORIG="test.csv"
SLICE="test_first100.csv"
head -n 1 $ORIG > $SLICE       # 헤더 복사
head -n 101 $ORIG | tail -n 100 >> $SLICE  # 데이터 100행 복사

BATCH_SIZE=10

for k in 2 3 4; do
  for lam in 0.7 0.8 0.9; do
    echo "▶ Running with MMR k=$k, λ=$lam (first 100 rows)"
    export MMR_K=$k
    export MMR_LAMBDA=$lam

    OUT="submission_k${k}_l${lam}_first100.csv"
    python main.py \
      --input_csv $SLICE \
      --output_csv $OUT \
      --batch_size $BATCH_SIZE

    echo "→ Completed: $OUT"
  done
done
