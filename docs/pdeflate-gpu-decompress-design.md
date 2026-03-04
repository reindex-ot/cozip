# PDeflate GPU解凍 実装設計

更新日: 2026-03-04

## 1. 目的

- PDeflate解凍を「CPU前処理 + GPU本処理 + CPU最終受け取り」に再設計する。
- 既存の同期待ち (`submit -> wait -> map`) をホットパスから外し、GPUの並列性を活かす。
- セクション独立性を前提に、VRAM固定長スロットで安定した高スループットを実現する。

## 2. 仕様前提（成立確認済み）

PDeflate v0 では以下が成立する。

1. 参照命令は「過去出力」ではなく「テーブルID参照」。
2. セクションは独立解凍可能。
3. セクション出力領域は `chunk_uncompressed_len` と `section_count` から決定式で再構成可能。
4. 各セクションの `cmd_len` は `section_index` の varint を読めば確定する。

これにより、解凍前に以下を確定できる。

- チャンク出力長 (`chunk_uncompressed_len`)
- セクション出力オフセット/長 (`out_offset/out_len`)
- セクションコマンド長/オフセット (`cmd_len/cmd_offset`)

## 3. 非目標

- Deflate一般形式のGPU解凍最適化（本設計はPDeflate専用）。
- 圧縮側の再設計（本設計は解凍のみ）。
- CPU/GPU比率の静的チューニング（まずはGPUパイプラインの同期構造を改善）。

## 4. 全体アーキテクチャ

## 4.1 CPU責務（薄く保つ）

- ヘッダ検証
- `section_index` から `cmd_offset/cmd_len` の復元
- `chunk_uncompressed_len` と `section_count` から `out_offset/out_len` 算出
- GPUジョブ記述子の作成
- 完了ジョブの最終出力バッファへの配置

CPUは中間データを展開しない。GPU実行中の待ちは最小化する。

## 4.2 GPU責務（本処理）

- セクション単位の命令デコード
- literalコピー
- table repeat参照展開
- 出力バッファへの書き込み

中間readbackは行わず、最終出力のみ回収する。

## 5. メモリ設計（固定長スロット）

GPU向けにリングバッファ方式を採用する。

## 5.1 スロット種別

- `cmd_slot`: コマンド列入力（可変長だが容量は固定上限）
- `section_meta_slot`: `cmd_offset/cmd_len/out_offset/out_len`（固定長）
- `table_slot`: `table_repeat`（固定長上限）
- `out_slot`: 展開出力（固定長、`max_chunk_uncompressed_len`）
- `error_slot`: エラーコード

## 5.2 固定長化ポリシー

- `out_slot` は `max_chunk_uncompressed_len` で固定確保。
- `section_meta_slot` は `max_section_count` 固定。
- `cmd_slot` は上限容量を設定し、超過チャンクは大容量スロット群へ振り分ける。
- 毎回 `create_buffer` しない。起動時または初回拡張時のみ確保。

## 5.3 アロケーション戦略

- 通常スロット群: 高頻度サイズ向け
- ラージスロット群: まれな大型チャンク向け
- フォールバック: どうしても収まらない場合のみCPU解凍

## 6. 実行モデル

## 6.1 マイクロバッチ + in-flight

- チャンクを小さなマイクロバッチでGPUに投入。
- ただし `submit/map` オーバーヘッド負けを避けるため、過小バッチは避ける。
- 複数バッチを同時in-flight化し、GPUを止めない。

## 6.2 回収モデル（重要）

- 一括Wait禁止。
- `collect-ready-any` で完了したスロットから回収。
- 回収順と出力順は分離し、最終出力は `chunk_index` で順序保証。

## 6.3 同期モデル

- `submit` と `map` をジョブ単位で直列化しない。
- `queue progress` は短いポーリングで進行確認し、readyジョブだけ map/copy。
- 未完了ジョブは次ループで再確認。

## 7. カーネル/キュー観点の最適化

## 7.1 カーネル

- 1セクション = 1ワーク単位を基本とする。
- セクション間同期を禁止。
- エラーはセクション単位コードで返し、CPU側で該当チャンクのみ処理を切り替える。

## 7.2 キュー

- `submit` をまとめる（小さいsubmit乱発を避ける）。
- CPU側で `wait -> map` を毎回実行しない。
- 可能な限り `copy -> map` を遅延回収化する。

## 8. 避けるべき設計

- 中間readback（命令長/部分出力）をホットパスへ入れる。
- GPUワーカー1本固定で都度ブロッキング回収する。
- 都度バッファ確保/解放を行う。
- チャンクごとに完全同期バリアを張る。

## 9. 計測設計（必須）

以下を分離して常時観測できるようにする。

- `submit_ms`
- `submit_done_wait_ms`
- `map_callback_wait_ms`
- `map_copy_ms`
- `kernel_timestamp_ms`（可能なら）
- `queue_stall_est_ms = submit_done_wait_ms - kernel_ms`
- `ready_any_hits` / `wait_loops`
- `inflight_batches` の平均/最大

これにより「カーネルが遅い」のか「待ち/回収が遅い」のかを確定できる。

## 10. 段階導入計画

1. フェーズ1: 固定長スロット + リング再利用
2. フェーズ2: ready-any回収 + 複数in-flight化
3. フェーズ3: submit/map待ちの詳細分離ログ
4. フェーズ4: マイクロバッチ自動調整（目標 in-flight 深さを維持）
5. フェーズ5: CPUフォールバック閾値の最小導入（最終手段）

## 11. 受け入れ基準

- CPUは前処理と最終配置に限定され、解凍ホットパスで長時間の単一スレッド待機が発生しない。
- `decomp_ms` が `decode_workers_ms` と整合し、外側I/Oコピー時間が混入しない。
- GPU解凍有効時に `map_wait` 支配が再発しない（再発時は原因メトリクスが特定可能）。
- 同一入力でCPU-onlyに対して安定した改善が得られる（少なくとも劣化が常態化しない）。

## 12. 補足

- セクション境界跨ぎ参照はPDeflateの意味論上不要。境界跨ぎを許可しない前提で実装する。
- テーブルエントリ内容が元データ上で境界を跨いでいても、解凍依存にはならない（read-only辞書参照のため）。
