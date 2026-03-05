# PDeflate GPU Async Batch Pipeline Design

## 0. この文書の目的

この文書は、PDeflate の GPU 経路を「同期回数最小化」で再設計するための実装設計である。

- 対象: `cozip_pdeflate` の GPU 圧縮/解凍経路
- 主眼: CPU/GPU 境界の待ち合わせを減らし、GPU 実行時間を隠蔽する
- 前提: **Hybrid スケジューラー（CPU/GPU の大域的割当ポリシー）は変更しない**

本設計は旧 `pdeflate-decode-v2-readback-ring-redesign.md` を置き換える。

---

## 1. 背景と現状ボトルネック

観測ログで一貫して以下が確認される。

- `kernel_timestamp_ms` に対して `submit_done_wait_ms` が大きい
- `queue_stall_est_ms` が大きい
- `map_copy_ms` は相対的に小さい
- 圧縮側では `hybrid-gpu-culprit` の `map_wait` 比率が高い

これは「PCIe 帯域不足」よりも、**同期レイテンシ（待ち合わせ回数と順序）**が支配していることを示す。

### 1.1 なぜ「2同期点」に見えるのに遅いか

高レベルでは「投げる→待つ→受け取る」の 1 往復に見えるが、実装上は 1 バッチの中で:

- 複数の `map_async`
- `poll(Poll)` + `poll(Wait)` の反復
- 複数 readback（サイズ/統計/payload）

が存在し、実効的には同期点が多段化している。

---

## 2. 設計原則

1. **スケジューラー非変更**
- `PDeflateHybridSchedulerPolicy` の方針や CPU/GPU 大域割当ロジックは変えない。
- 最適化対象は GPU 実行経路内部（prepare/submit/complete）。

2. **同期はゼロにできない。回数を減らす**
- CPU が最終結果を使う以上、同期点は必要。
- 目的は「同期点の削減」と「同期を終端に寄せる」。

3. **1チャンク同期禁止、バッチ単位同期**
- チャンクごとに待たない。
- 小バッチ（複数 chunk）を 1 実行単位として扱う。

4. **submit/complete 分離**
- submit 側は待たない。
- complete 側が ready をまとめて回収。

5. **可観測性の組み込み**
- kernel/copy/wait を常に分離記録し、回帰を追えるようにする。

---

## 3. 非目標（今回やらないこと）

- Hybrid の CPU/GPU 仕事配分ルールの全面変更
- 圧縮形式やビットストリーム仕様変更
- カーネルアルゴリズムの大幅変更（Huffman 処理最適化など）

---

## 4. 新アーキテクチャ（スケジューラー不変更）

## 4.1 3ステージ非同期パイプライン

同一 GPU worker 内を論理的に 3 ステージに分離する。

1. `Prepare`
- CPU 前処理で複数 chunk をバッチ化
- GPU 入力バッファへ pack

2. `Submit`
- GPU へ dispatch / copy submit
- 完了待ちはしない
- inflight 上限（クレジット）だけ守る

3. `Complete`
- 完了済みバッチだけ回収
- readback/map をまとめて実行
- CPU 後処理へ引き渡す

### 4.2 実行モデル

1 ループで以下を回す:

1. `Complete` を先に回してクレジット回復  
2. クレジットが空くまで `Submit`  
3. `Prepare` を先行生成（次バッチ）

`Prepare` と `Complete` を交互に進めることで、待ちを隠蔽する。

重要:

- `Submit` 直後に `Wait` へ入らない。
- GPU 完了待ちより先に、次バッチの `Prepare` を進める。
- CPU 前処理と GPU 実行を重ねることで、境界待ち時間の露出を減らす。
- `Complete` で回収した直後も、可能なら CPU 後処理より先に次バッチを `Submit` する。
- 目的は「GPU を遊ばせない」ことであり、後処理は `Submit` 後に追随させる。

---

## 5. データ構造

```text
PreparedBatchQueue   : VecDeque<PreparedBatch>
InflightBatchQueue   : VecDeque<InflightBatch>
CompletedBatchQueue  : VecDeque<CompletedBatch>
CreditPool           : { max_inflight_batches, max_inflight_bytes }
```

## 5.1 PreparedBatch

- `batch_id`
- `chunk_indices[]`
- `gpu_input_offsets[]`
- `expected_output_sizes[]`
- `resource_handles`（入力/メタバッファ）

## 5.2 InflightBatch

- `submit_seq`
- `submitted_at`
- `readback_plan`
- `map_receivers[]`
- `kernel_timestamp_query_handle`（利用可能なら）

## 5.3 CompletedBatch

- `chunk_results[]`
- `copy_ms`
- `map_wait_ms`
- `kernel_ms`
- `fallback_flags[]`

---

## 6. 同期削減の具体策

## 6.1 ループ内 `Wait` の抑制

- 既定は `poll(Poll)` を複数回
- `poll(Wait)` は以下の時のみ:
  - inflight が高水位到達
  - 連続で進捗ゼロ
- `Submit` 成功後は `Wait` より先に `Prepare` を回し、前処理を先食いする

## 6.2 回収の集約

- map/readback は「完了した batch 単位」でまとめる
- 1 chunk 完了ごとの回収を禁止
- 完了回収後の処理順は `Submit` 優先、CPU 後処理はその直後に実行する

## 6.3 copy 計画の固定化

- バッチ内出力配置を固定し、回収時の CPU 側 scatter を最小化

## 6.4 map timeout の局所化

- timeout は batch 単位で扱う
- timeout 発生時は当該 batch のみ fallback
- 全体 drain/fallback はしない

---

## 7. 小バッチの定義

小バッチは「複数 chunk を 1 単位に束ねて同期コストを按分する単位」。

単に連続 submit するだけでは小バッチではない。必要要件:

- 入力 pack が 1 まとまり
- dispatch が少数回
- 完了回収が batch 単位
- CPU 後処理が batch 単位

推奨初期値:

- `batch_chunks`: 8〜32
- `max_inflight_batches`: 2〜6（VRAM と map 遅延を見て調整）

---

## 8. 制御パラメータ（実装時）

- `GPU_PIPE_TARGET_INFLIGHT_BATCHES`
- `GPU_PIPE_WAIT_HIGH_WATERMARK`
- `GPU_PIPE_SPIN_POLLS_BEFORE_WAIT`
- `GPU_PIPE_MAP_TIMEOUT_MS`
- `GPU_PIPE_MAX_INFLIGHT_BYTES`

原則:

- まず inflight を維持して GPU 空転を減らす
- ただし map 待ち増大時は高水位を下げる

---

## 9. 計測項目（必須）

既存ログに加えて以下を追加する。

- `pipe_prepare_ms`
- `pipe_submit_ms`
- `pipe_complete_ms`
- `pipe_inflight_batches_avg/max`
- `pipe_inflight_bytes_avg/max`
- `pipe_wait_poll_ms`
- `pipe_wait_block_ms`
- `pipe_map_wait_ms`
- `pipe_map_copy_ms`
- `pipe_kernel_ms`
- `pipe_queue_stall_est_ms`
- `pipe_fallback_batches{timeout,map_err,collect_err}`

成功判定は throughput 単体でなく、以下を併用する:

- `queue_stall_est_ms / kernel_ms`
- `wait_block_ms / total_gpu_call_ms`
- `writer_hol_wait_ms`

---

## 10. 実装フェーズ

## Phase A1: 既存経路のパイプライン化（仕様温存）

- 既存関数内部を `Prepare/Submit/Complete` の3段に分離
- 出力仕様は不変
- chunk 単位同期を batch 単位へ寄せる

完了条件:

- 既存テスト全通
- throughput 回帰なし
- `wait_block_ms` 減少

## Phase A2: 回収集約の強化

- readback/map を batch 集約
- 完了回収の ready-any 処理を明示化

完了条件:

- `queue_stall_est_ms/kernel_ms` 改善
- `map_wait_ms` の分散縮小

## Phase A3: 自動調整

- inflight と wait 閾値を軽量適応制御
- profile off 時のオーバーヘッド抑制

完了条件:

- `runs=3~5` の中央値で安定改善

---

## 11. 失敗時フォールバック戦略

1. batch 単位で CPU fallback（局所）
2. 同一理由の連続失敗が閾値超過で GPU 経路を短時間サーキットブレーク
3. 次バッチ以降で再試行

「全体停止」「全バッチ一括フォールバック」は避ける。

---

## 12. 検証手順（運用コマンド）

## 12.1 速度比較（低ノイズ）

```bash
COZIP_PDEFLATE_GPU_DECODE_BATCH_CHUNKS=16 \
target/release/examples/bench_pdeflate \
  --size-mib 8000 --mode speed --runs 5 --warmups 1 \
  --chunk-mib 4 --sections 128 \
  --gpu-compress --compare-hybrid --dataset bench \
  --gpu-slot-count 6 --gpu-batch-chunks 6 --gpu-submit-chunks 4 \
  --no-skip-decompress --no-verify
```

## 12.2 待ち内訳確認

```bash
COZIP_PDEFLATE_GPU_DECODE_BATCH_CHUNKS=16 \
COZIP_PDEFLATE_PROFILE=1 \
COZIP_PDEFLATE_PROFILE_GPU_DECODE_V2_WAIT_PROBE=1 \
target/release/examples/bench_pdeflate \
  --size-mib 8000 --mode speed --runs 1 --warmups 1 \
  --chunk-mib 4 --sections 128 \
  --gpu-compress --compare-hybrid --dataset bench \
  --gpu-slot-count 6 --gpu-batch-chunks 6 --gpu-submit-chunks 4 \
  --no-skip-decompress --no-verify
```

---

## 13. 受け入れ基準

最低基準:

- `speedup_decomp(cpu/hybrid) >= 1.00` を runs=5 中央値で達成
- `queue_stall_est_ms / kernel_ms` を現状比 20% 以上削減

目標基準:

- `speedup_decomp >= 1.10`（中央値）
- `writer_hol_wait_ms` を現状比 30% 以上削減

---

## 14. 要点まとめ

- 現在の支配項はカーネル計算より「同期段取り」
- 勝ち筋は「スケジューラー変更」ではなく、内部の同期回数削減と batch 完了回収
- 3ステージ（Prepare/Submit/Complete）を明示化し、待ちを終端へ押し込む
- fallback は局所化し、全体停止を避ける
