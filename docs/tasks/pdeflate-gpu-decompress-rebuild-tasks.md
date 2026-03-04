# PDeflate GPU解凍 再実装タスク分解

更新日: 2026-03-04  
元設計: `docs/pdeflate-gpu-decompress-design.md`

## 0. 前提

- 本タスクは「改修」ではなく「再実装」を前提とする。
- 既存GPU解凍実装は温存しない。まず削除し、最小構成から作り直す。
- 途中段階で動作を維持するため、削除直後はCPU解凍フォールバックを明示する。

## 1. マイルストーン

- M1: 既存GPU解凍実装の除去完了（CPUのみで正常動作）
- M2: 新GPU解凍の最小動作（単一バッチ、正しさ優先）
- M3: in-flight + ready-any 回収実装
- M4: 固定長スロット再利用と計測分離完了
- M5: 性能評価と受け入れ基準達成

## 2. タスク一覧

## T00 既存GPU解凍実装の削除（必須）

- [x] `legacy_pdeflate_cpu/gpu.rs` の decode 用シェーダ/パイプライン/バインド群を削除
- [x] `decode_sections_chunk_gpu` とその依存コードを削除
- [x] 解凍経路から旧GPU decode 呼び出しを除去し、CPU decode のみで通す
- [x] 旧GPU decode専用ログ/メトリクスを一旦削除
- [x] `cargo check` とベンチ実行で「CPU decodeのみ」で回ることを確認

完了条件:
- 旧GPU decode関連コードがビルド対象から完全に消えている
- 解凍の正しさがCPU経路で維持される

## T01 新GPU解凍 I/F の骨組み作成

- [x] `GpuDecodeJob` / `GpuDecodeResult` / `GpuDecodeSlot` を新規定義
- [x] CPU前処理で必要情報を確定:
- `chunk_uncompressed_len`
- `cmd_offset/cmd_len`
- `out_offset/out_len`
- [x] 新API（例: `decode_chunks_gpu_v2(...)`）を追加
- [x] 旧I/F互換の呼び出し口を暫定で新APIへ接続

完了条件:
- ダミー実装でも新I/F経由でCPU fallbackが動作する

## T02 固定長VRAMスロット（リング）実装

- [x] `cmd_slot` / `section_meta_slot` / `table_slot` / `out_slot` / `error_slot` を実装
- [x] スロット再利用管理（free/inflight/ready）を実装
- [x] 通常スロットと大型スロットの2系統を用意
- [x] 毎回 `create_buffer` しない設計へ置換

完了条件:
- 連続実行してもGPUメモリ確保回数が初期化時以外ほぼ増えない

## T03 新GPU decodeカーネル最小版

- [x] セクション独立デコードカーネルを新規実装
- [x] エラーコード設計（section単位）を実装
- [x] CPU側でエラーチャンクのみフォールバックできる経路を作成
- [x] CPU/GPU一致テスト（ランダム入力 + 固定種）を追加

完了条件:
- GPU decode最小版で正しい出力を返せる

## T04 in-flight投入と ready-any 回収

- [x] submitをマイクロバッチ単位で継続投入
- [x] 完了回収を `collect-ready-any` 化
- [x] 一括Wait経路を削除
- [x] out-of-order完了を `chunk_index` で順序整列

完了条件:
- 複数バッチを同時in-flightで維持できる
- 完了順が前後しても最終出力順は正しい

## T05 map/readbackのホットパス外し

- [x] 中間readbackを禁止（最終出力のみ）
- [x] map待ちの局所化（readyジョブのみmap）
- [x] submit/mapの直列化を解消

完了条件:
- 中間readbackを伴うコードが解凍ホットパスに存在しない

## T06 計測ログ再構築

- [x] 以下を分離ログ化:
- `submit_ms`
- `submit_done_wait_ms`
- `map_callback_wait_ms`
- `map_copy_ms`
- `kernel_timestamp_ms`（可能なら）
- `queue_stall_est_ms`
- `ready_any_hits`
- `inflight_batches(avg/max)`
- [x] 旧ログとの互換が必要なものは名称移行を記録
- 名称移行メモ:
- 旧: `chunk-decode-gpu`（旧GPU decode実装）  
  新: `decode-v2-wait-breakdown`（ジョブ単位） / `decode-v2-ready-any`（集計）

完了条件:
- 「カーネル遅延」と「同期待ち遅延」をログだけで区別できる

## T07 テスト整備

- [x] セクション境界ケース（最小/最大/端数）テスト
- [x] 大小チャンク混在テスト（通常/大型スロット）
- [x] GPU失敗時の局所フォールバックテスト
- [x] 順序保証テスト（out-of-order完了入力）

完了条件:
- GPU有無で同一入力の出力一致を継続検証できる

## T08 性能検証と受け入れ判定

- [ ] 4GiB/8GiBで CPU_ONLY vs CPU+GPU(Hybrid) を比較
- [ ] `decomp_ms` と内訳の整合性確認
- [ ] 単一スレッド長時間張り付きの再発有無確認
- [ ] map_wait再支配の有無確認

受け入れ基準:
- `decomp_ms` に不要な外側I/O待ちが混入しない
- 旧実装比で待ち時間支配が緩和される
- 少なくとも特定条件でCPU_ONLYに対し有意な改善が再現する

## 3. 実装順（推奨）

1. T00
2. T01
3. T02
4. T03
5. T04
6. T05
7. T06
8. T07
9. T08

## 4. リスクとガードレール

- リスク: 全削除後の長期不安定化
- 対策:
- T00完了直後にCPU-onlyの安定動作を固定
- T03まで機能フラグで新旧を切替可能にして短サイクル検証
- 1タスクごとに `cargo check` + 最小ベンチを必須化

## 5. 進捗記録テンプレート

各タスク完了時に追記すること:

- 実施日:
- 実施タスク:
- 変更ファイル:
- 検証コマンド:
- 結果:
- 残課題:
