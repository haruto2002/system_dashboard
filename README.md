# system_dashboard

群衆解析（Crowd Analysis）のリアルタイム処理システムから出力される追跡データを入力として、
**軌跡・混雑リスク・時系列グラフ・注目領域**を可視化するダッシュボード生成ツールです。

横浜（`yokohama_2020508`）の複数エリア（`worldporter` / `akarenga` / `chosha` / `kokusaibashi`）を対象に、
各フレームごとの JSON データを読み込み、可視化画像を生成・保存します。

---

## 📌 このリポジトリで取り組むタスク（担当者向け）

このリポジトリを引き継いだら、以下の順で進めてください。詳細は各セクションを参照。

- [ ] **1. コード理解**
  - 本 README の [アーキテクチャ](#アーキテクチャ) と [各コンポーネントの役割](#各コンポーネントの役割) を読む
  - 実際に `main.py` を動かして、`tmp/frame_XXXXXX/` に出力される各画像を確認する
  - `dashboard.display()` の処理フローを追い、「入力 JSON → 各可視化画像」までのデータの流れを掴む
- [ ] **2. 最高なダッシュボードに仕上げる**
  - [ ] **各パーツの可視化を綺麗にする**（軌跡 / ヒートマップ / グラフ / 注目領域 の見た目改善）
  - [ ] **それぞれの可視化を組み合わせて一枚に統合**（`main.py` の `create_dashboard_image()` を実装）

詳しい作業指針は [担当者へのタスク詳細](#担当者へのタスク詳細) を参照してください。

---

## アーキテクチャ

```
main.py
  └─ Dashboard (vis_tool/dashboard.py)   ← 各コンポーネントを束ねる司令塔
       ├─ TrackPreparer      (components/utils.py)      … 生JSON → 追跡データ（マップ座標へ射影）
       ├─ MapDisplayer       (components/trajectory.py) … 全体マップ上に軌跡/矩形を描画・エリア切り出し
       ├─ CrowdRiskScore     (components/risk.py)       … 混雑リスクスコア(CRS)をグリッドで計算
       ├─ HeatmapDisplayer   (components/risk_vis.py)   … リスクスコアをヒートマップ化して重畳
       ├─ GraphGenerator     (components/graph.py)      … 人数・リスクスコアの時系列グラフを生成
       ├─ FocusAreaSelector  (components/focus.py)      … リスクの高い注目領域を Top-K 抽出
       └─ FocusAreaDrawer    (components/focus.py)      … 注目領域を各画像に矩形描画・切り出し
```

### 処理フロー（1フレームあたり）

`Dashboard.display(data, timestamp)` が1フレーム分の入力を受け取り、以下を順に実行します。

1. **追跡データ整形** — `TrackPreparer` が JSON を読み、ホモグラフィ変換で各点を全体マップ座標へ射影
2. **軌跡描画** — `MapDisplayer` が全体マップに軌跡・矩形を描画し、エリアごとに切り出し
3. **リスク計算** — `CrowdRiskScore` がエリアごとにグリッド状の混雑リスクスコアマップを計算
4. **ヒートマップ描画** — `HeatmapDisplayer` がリスクマップを軌跡画像に重畳
5. **グラフ更新/生成** — `GraphGenerator` が人数とリスクスコアの時系列を蓄積し折れ線グラフ化
6. **注目領域抽出** — 一定フレームごと（`focus_area_update_frequency=3`）に累積リスクから Top-K 領域を抽出
7. **注目領域描画** — 全体画像・各エリア画像に注目領域の矩形を描画

戻り値:

```python
trajectory_img, heatmap_img_dict, focus_area_outputs, graph_img_dict = dashboard.display(data, timestamp)
```


| 戻り値                  | 型                                     | 内容                       |
| -------------------- | ------------------------------------- | ------------------------ |
| `trajectory_img`     | `np.ndarray`                          | 全体マップ上の軌跡＋注目領域           |
| `heatmap_img_dict`   | `dict[str, np.ndarray]`               | エリア名 → リスクヒートマップ重畳画像     |
| `focus_area_outputs` | `list[tuple[str, np.ndarray, float]]` | (エリア名, 注目領域の切り出し画像, スコア) |
| `graph_img_dict`     | `dict[str, np.ndarray]`               | エリア名 → 人数・リスクの時系列グラフ     |


---

## 各コンポーネントの役割

### `TrackPreparer`（`components/utils.py`）

- 入力 JSON の `point`（人物中心点）と `bbox`（矩形）を ID ごとにまとめる
- `MapConfig.to_map_coords()` でホモグラフィ変換を行い、カメラ座標 → 全体マップ座標へ射影
- 最新フレームに存在する ID のみを追跡対象とする

### `MapDisplayer`（`components/trajectory.py`）

- 全体マップ画像（`all_map.jpg`）を背景に、軌跡の点列と進行方向を示す回転矩形を描画
- `map_draw_style.yaml` で描画スケール・点/矩形/枠線のスタイルを設定
- `crop_places()` で全体画像から各エリアの領域を切り出す

### `CrowdRiskScore`（`components/risk.py`）

- グリッドごとに **混雑リスクスコア (Crowd Risk Score)** を計算
- 局所密度（ガウシアンカーネル）× 速度ベクトルの発散などから、対向・衝突リスクを評価
- モードは `crs` / `crs+` / `crs++` の3種（現状は `crs++`）

### `HeatmapDisplayer`（`components/risk_vis.py`）

- リスクスコアマップを正規化し `COLORMAP_JET` でヒートマップ化、背景画像に半透明重畳
- ベクトル矢印の描画機能（`draw_vec`）もあるが現状は未使用

### `GraphGenerator`（`components/graph.py`）

- フレームごとに「人数」「リスクスコア合計」を蓄積し、左右2軸の時系列折れ線グラフを生成
- `matplotlib` で描画し、OpenCV 画像（`np.ndarray`）として返す
- ※`NumPeopleGraphGenerator` は人数のみの別実装（参考/実験用）

### `FocusAreaSelector` / `FocusAreaDrawer`（`components/focus.py`）

- 累積リスクマップから矩形和が最大の領域を Top-K 抽出（NMS で重複抑制）
- 抽出した注目領域を全体画像・各エリア画像に色付き矩形で描画し、切り出し画像も返す

---

## ディレクトリ構成

```
system_dashboard/
├─ main.py                     # エントリポイント（フレームを回して各画像を保存）
├─ pyproject.toml              # 依存関係（uv 管理）
├─ uv.lock
├─ received_synced_data/       # 入力JSON群（.gitignore 対象）
├─ tmp/                        # 出力画像（.gitignore 対象。frame_XXXXXX/ ごとに保存）
└─ vis_tool/
   ├─ dashboard.py             # Dashboard クラス
   ├─ components/              # 各可視化コンポーネント
   │  ├─ utils.py
   │  ├─ trajectory.py
   │  ├─ risk.py
   │  ├─ risk_vis.py
   │  ├─ graph.py
   │  └─ focus.py
   └─ map_data/yokohama_2020508/
      ├─ all_map.jpg           # 全体マップ画像
      ├─ map_draw_style.yaml   # 描画スタイル設定
      └─ map_config/           # エリアごとのホモグラフィ・座標設定
         ├─ worldporter.yaml
         ├─ akarenga.yaml
         ├─ chosha.yaml
         └─ kokusaibashi.yaml
```

---

## セットアップ

依存管理は [uv](https://github.com/astral-sh/uv) を利用します（`pyproject.toml` / `uv.lock`）。

```bash
# 依存関係のインストール
uv sync

# 実行
uv run python main.py
```

Python は `>=3.12.8,<3.13` が必要です。

### 事前準備

**入力データ** `received_synced_data/*.json` を配置する
  - ファイル名は `連番_タイムスタンプ.json`（例: `000001_902092.json`）
  - JSON 形式:
    ```json
    {
      "<place>": [
        {
          "camera_id": ..., "pc_id": ..., "timestamp": ..., "frame_id": ...,
          "objects": {
            "point": [{"id": 1, "x": 6149.7, "y": 98.2}, ...],
            "bbox":  [{"id": 1, "x1": ..., "y1": ..., "x2": ..., "y2": ...}, ...]
          }
        }
      ]
    }
    ```

---

## 出力

`main.py` を実行すると、`tmp/frame_XXXXXX/` 以下に各フレームの可視化画像が保存されます。

```
tmp/frame_000001/
├─ trajectory_img.jpg          # 全体マップ＋軌跡＋注目領域
├─ heatmap_img_<place>.jpg     # エリアごとのリスクヒートマップ
├─ graph_<place>.jpg           # エリアごとの時系列グラフ
└─ topNN.jpg                   # 注目領域の切り出し画像
```

> 現状 `main.py` は動作確認のため `if i > 5: break` で **先頭6フレームのみ** 処理します。

---

## 酒井くんにやって欲しいこと

### 1. コード理解

- まず `uv run python main.py` を実行し、`tmp/` に出る画像を眺めて全体像を掴む
- `Dashboard.display()` を起点に、上記 [処理フロー](#処理フロー1フレームあたり) を実際のコードで追う
- 各 `*.yaml`（`map_config` / `map_draw_style`）のパラメータが描画にどう効くか確認する

### 2-a. 各パーツの可視化を綺麗にする

もっと見やすくできそうであれば積極的に改善してください。

### 2-b. それぞれの可視化を一枚に統合（メイン）

現状 `main.py` の `create_dashboard_image()` は **未実装（`pass`）** です。ここを実装し、
`display()` の4つの戻り値（軌跡・ヒートマップ群・注目領域群・グラフ群）を
**1枚のダッシュボード画像**にレイアウト統合してください。

```12:12:main.py
def create_dashboard_image():
```

```52:54:main.py
        # 各生成物を一つの画像として保存
        # dashboard_img = create_dashboard_image()
        # cv2.imwrite(save_dir / "dashboard.jpg", dashboard_img)
```

実装のヒント:

- 全体軌跡を大きく中央に、周囲に各エリアのヒートマップ・グラフ・注目領域を配置する等
- 画像サイズを揃えるヘルパー（リサイズ・余白パディング）を用意すると組みやすい
- `create_dashboard_image()` に必要な戻り値を引数として渡す形にリファクタする
- 完成したら `main.py` のコメントアウトを解除し、`tmp/dashboard.jpg` として保存する

