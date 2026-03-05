# PDeflate v0 ベースライン（実装前固定）

最終更新: 2026-03-05

## 目的
- T00（前提固定）で使用する比較基準を固定する。
- 以後の変更は本ファイルのコマンド条件と比較する。

## 固定ベンチコマンド

```bash
COZIP_PDEFLATE_PROFILE=1 \
COZIP_PDEFLATE_PROFILE_DETAIL=0 \
COZIP_PDEFLATE_PROFILE_GPU_DECODE_V2_WAIT_PROBE=1 \
target/release/examples/bench_pdeflate \
  --size-mib 8000 \
  --mode speed \
  --runs 1 \
  --warmups 1 \
  --chunk-mib 4 \
  --sections 128 \
  --gpu-compress \
  --compare-hybrid \
  --dataset bench \
  --gpu-slot-count 6 \
  --gpu-batch-chunks 6 \
  --gpu-submit-chunks 4 \
  --no-skip-decompress \
  --no-verify
```

## 比較用ログ（要約）

入力ログ日付: 2026-03-05

- `gpu_adapter`: NVIDIA GeForce RTX 5070 Laptop GPU (Vulkan)
- `size_mib=8000`, `chunk_mib=4`, `sections=128`

### 実行結果サマリ
- `CPU_ONLY`: `comp_ms=4305.424`, `decomp_ms=651.940`
- `CPU+GPU`: `comp_ms=4293.155`, `decomp_ms=692.491`
- `ratio=0.3951`
- `speedup_comp(cpu/hybrid)=1.003x`
- `speedup_decomp(cpu/hybrid)=0.941x`

### 解凍配分（Hybrid）
- `cpu_claimed_chunks=1956`
- `gpu_claimed_chunks=44`
- `gpu_busy_ms=681.183`

## 備考
- 本ベースラインは「4-byte section境界仕様変更後」の比較基準。
- 以降の最適化（T01以降）は、本ファイルの値に対して改善/悪化を判定する。
