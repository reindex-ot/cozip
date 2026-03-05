# PDeflate v0（Huffman LUT）実装タスク分解

対象仕様:
- `docs/pdeflate-v0-spec.md`

方針:
- チャンクごとに `Huffman LUT (root table + subtable)` を保持
- CPU解凍: チャンク並列
- GPU解凍: チャンク + セクション並列

---

## T00: 前提固定とブランチ初期化
- [x] 現在のベンチコマンドを固定し、比較用ログを保存する（`docs/pdeflate-v0-baseline.md`）
- [x] 互換性ポリシーを明文化する（v0新形式専用、旧形式は明示エラー）
- [x] `docs/pdeflate-v0-spec.md` を唯一の仕様ソースとして参照する旨をREADMEへ追記

完了条件:
- 実装前ベースライン（圧縮率/圧縮速度/解凍速度）が再現できる

---

## T01: フォーマット定義の更新（ヘッダ・オフセット）
- [x] チャンクヘッダを新仕様へ更新
  - `huff_lut_offset`
  - `section_bitstream_offset`
- [x] オフセット検証ロジックを更新
- [x] `section_index` の意味を `cmd_len` から `bit_len` に変更

完了条件:
- 新ヘッダで roundtrip が通る（最小ケース）
- 破損データで妥当なエラーを返す

---

## T02: 論理命令列 -> ビットストリーム層の追加
- [x] 既存の論理命令列（TABLE_REF/LITERAL_RUN）を Huffman 対象シンボル列へ変換
- [x] セクションごとに bitstream を生成
- [x] `section_index` には各セクション `bit_len`（ULEB128）を書き込む

完了条件:
- [x] 論理命令列とビットストリームの相互変換テストが通る

---

## T03: Huffmanコード生成とLUT生成
- [x] Canonical Huffman コード生成を実装（固定/動的は実装方針に従う）
- [x] `root table + subtable` 形式の高速LUTを構築
- [x] LUTシリアライズ/デシリアライズを実装

完了条件:
- [x] 同一入力で CPU LUT復号が木走査復号と一致する
- [x] 不正LUT（範囲外・循環相当）を拒否できる

---

## T04: 圧縮（CPU）パスの新フォーマット化
- [x] チャンク生成時に Huffman LUT ブロックを書き込む
- [x] セクションbitstreamを `section_bitstream_offset` 以降へ連結
- [x] `table_index/table_data` と新ブロックの整合を保つ

完了条件:
- [x] 新フォーマットのみで圧縮 -> 解凍 roundtrip が通る
- [x] 既存ベンチが動作する

---

## T05: 解凍（CPU）パスの更新
- [x] 新フォーマットパーサー（LUT + section bitstream）を実装
- [x] チャンク並列ワーカーで各チャンクを独立解凍
- [x] セクション `bit_offset/bit_len` の prefix-sum 再構築を実装

完了条件:
- [x] `CPU_ONLY` で新フォーマットの全テスト通過
- [x] プロファイルログで `decompress-total` が正しく出る

---

## T06: GPU解凍カーネル（LUT復号）
- [x] GPU側メタデータへ LUT 参照情報を追加
- [x] セクション単位復号カーネルで `root + subtable` を使用
- [x] セクションごとに `out_offset/out_len` へ独立書き込み

完了条件:
- [x] GPU復号結果がCPU復号結果と一致
- [x] セクション並列時に境界破壊がない

---

## T07: Hybridスケジューラ調整
- [ ] チャンクキュー配分（CPU/GPU）を新フォーマットに合わせて調整
- [ ] GPU投入粒度（batch/in-flight）の規定値を見直し
- [ ] CPUフォールバック発生時の扱いを統一（許容時はログ、非許容時はエラー）

完了条件:
- `decompress-scheduler-claims` でCPU/GPUの配分が観測可能
- フォールバック動作が仕様どおり

---

## T08: バリデーション強化
- [ ] `sum(section_bit_len)` と実ビット長の一致検証
- [ ] LUT参照範囲検証（root/subtable）
- [ ] 4-byte内部境界と `out_len` 一致検証

完了条件:
- 破損ケーステストで即エラー
- OOB/無限ループ系の再現ケースをブロック

---

## T09: 計測項目の拡張
- [ ] decodeログに以下を追加
  - `kernel_ms`（純カーネル）
  - `queue_stall_ms`
  - `map/readback_ms`
  - `lut_load_ms`
  - `section_decode_ms`
- [ ] CPU/GPUの `ms/chunk` を継続出力

完了条件:
- ボトルネックが「計算」「待ち」「転送」で切り分け可能

---

## T10: 互換性モード（必要時）
- [ ] 旧形式読み取りを残す場合は `flags/version` で分岐
- [ ] 旧形式廃止の場合はエラーメッセージを明確化

完了条件:
- どちらの方針でもユーザーが原因を判別できる

---

## T11: テスト拡充
- [ ] 単体テスト
  - LUT生成/復号
  - section bit境界
  - 4-byte内部境界
- [ ] 結合テスト
  - CPU_ONLY roundtrip
  - CPU+GPU roundtrip
  - 大小チャンク混在
- [ ] 破損入力テスト
  - offsets不整合
  - bit_len不整合
  - LUT不正

完了条件:
- CI想定のテストセットが安定通過

---

## T12: ベンチ・チューニング
- [ ] 主要パラメータ探索
  - `gpu_batch_chunks`
  - `gpu_submit_chunks`
  - `in-flight`
  - `section_count`
- [ ] 指標比較
  - 圧縮率
  - CPU_ONLY解凍
  - Hybrid解凍
- [ ] 期待値とのギャップを記録し、次アクション化

完了条件:
- ベンチ結果に基づく推奨デフォルト値を提示

---

## 実行順（推奨）
1. T00-T03（フォーマットと符号化基盤）
2. T04-T06（圧縮/解凍本体）
3. T07-T09（運用性と可観測性）
4. T10-T12（互換性・品質・性能仕上げ）
