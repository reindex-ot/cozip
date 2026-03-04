# Tasks

このディレクトリは、設計書を「実装順の作業項目」に分解したタスク管理用ドキュメントを置く。

## 一覧

- `pdeflate-gpu-decompress-rebuild-tasks.md`
  - 元設計: `docs/pdeflate-gpu-decompress-design.md`
  - 目的: PDeflate GPU解凍の再実装タスク分解
  - 方針: 既存GPU解凍実装を一度削除し、固定長スロット + in-flight + ready-any 回収で再構築
