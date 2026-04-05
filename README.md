# video-keyframe-extract

批量影片關鍵幀擷取工具。針對大量短影片（10–30 秒街景類），自動偵測場景切換與語意變化，擷取代表性關鍵幀。

## 功能

- **硬剪輯偵測**：PySceneDetect 精準偵測蒙太奇式場景跳變點
- **語意差異取樣**：CLIP ViT-B/32 計算幀間語意距離，畫面有實質變化時觸發擷取
- **最大間隔保底**：連續影片超過設定秒數未擷取，強制捕捉一張
- **每片上限保護**：每部影片最多擷取 N 張，防止縮時影片過度擷取

## 需求

- Docker（需支援 NVIDIA GPU）
- NVIDIA GPU（已測試 RTX 3060 12GB）

## 快速開始

```bash
# Build
docker build -t video-extract:latest .

# 小批量測試 (前 5 部)
docker run --rm -it --gpus=all --ipc=host -v "${PWD}/:/app" video-extract:latest \
    python extract.py --limit 5

# 全量執行
docker run --rm -it --gpus=all --ipc=host -v "${PWD}/:/app" video-extract:latest \
    python extract.py
```

## 目錄結構

```
├── Dockerfile
├── requirements.txt
├── extract.py          # 主程式
├── video/              # 輸入影片
└── output/             # 輸出圖片 ({影片序號}_{圖片序號}.jpg)
```

## 輸出規則

- 影片按檔名字母順序排列，序號 3 位數 zero-padded（`001`, `002`, ...）
- 圖片序號 3 位數 zero-padded（`001_001.jpg`, `001_002.jpg`, ...）

## 可調參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--cut-threshold` | `27.0` | PySceneDetect 硬切偵測靈敏度（越低越敏感） |
| `--semantic-threshold` | `0.12` | CLIP 語意距離閾值（越低越容易觸發擷取） |
| `--sample-fps` | `2.0` | 取樣頻率（每秒幾幀） |
| `--max-interval` | `6.0` | 最大間隔保底（秒），超過則強制擷取 |
| `--max-captures` | `15` | 每部影片最多擷取張數 |
| `--jpeg-quality` | `95` | 輸出 JPEG 品質 |
| `--limit` | `0` | 只處理前 N 部影片（0 = 全部） |

### 調參範例

```bash
# 擷取太少 → 降低語意閾值
python extract.py --semantic-threshold 0.08

# 擷取太多 → 提高語意閾值
python extract.py --semantic-threshold 0.18

# 一鏡到底影片漏拍 → 縮短最大間隔
python extract.py --max-interval 3.0
```
