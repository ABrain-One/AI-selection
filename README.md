AI Selection

## Create and Activate a Virtual Environment (recommended)
For Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For Windows:
```bash
python3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Usage Examples (ranked by the accuracy–time trade-off, no weight to choose):

MAI 2026 INT8, SSIM vs inference time ms (NPU): https://arxiv.org/pdf/2611.00963
```bash
python -m ab.nn.select --rank koreatechV2,0.8499,4.35 Godzilla,0.8831,215160 AntSR,0.8731,4.52 LoongSR,0.8586,5729 IN2GM,0.8701,46.5 z6,0.8699,5.49 AIO_MAI,0.8634,41.1 AntYSP,0.8729,4.33
```

MAI 2026 PSNR (conversion by provided formula to the range 0 - 1) vs inference time ms (NPU): https://arxiv.org/pdf/2611.00963
```bash
python -m ab.nn.select --rank koreatechV2,0.998758348,4.35 Godzilla,0.999085887,215160 AntSR,0.998995384,4.52 LoongSR,0.998943182,5729 IN2GM,0.998964858,46.5 z6,0.998983751,5.49 AIO_MAI,0.998950458,41.1 AntYSP,0.998990747,4.33
```

EfficientNet B0–B7, ImageNet top-1 vs FLOPs (B): https://arxiv.org/pdf/1905.11946, Table 2
```bash
python -m ab.nn.select --rank B7,0.843000000,37 B0,0.771000000,0.39 B4,0.829000000,4.2 B6,0.840000000,19 B2,0.801000000,1 B1,0.791000000,0.7 B3,0.816000000,1.8 B5,0.836000000,9.9
```

EfficientNet B0–B7, top-1 vs params (M): https://arxiv.org/pdf/1905.11946, Table 2
```bash
python -m ab.nn.select --rank B5,0.836000000,30 B2,0.801000000,9.2 B1,0.791000000,7.8 B4,0.829000000,19 B7,0.843000000,66 B3,0.816000000,12 B0,0.771000000,5.3 B6,0.840000000,43
```

YOLO11, mAP vs T4 TensorRT10 latency (ms): https://docs.ultralytics.com/models/yolo11/
```bash
python -m ab.nn.select --rank YOLO11l,0.534000000,6.2 YOLO11m,0.515000000,4.7 YOLO11x,0.547000000,11.3 YOLO11n,0.395000000,1.5 YOLO11s,0.470000000,2.5
```

MAI 2021 camera scene detection, Top-1 vs A11 Bionic runtime: https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Ignatov_Fast_and_Accurate_Quantized_Camera_Scene_Detection_on_Smartphones_Mobile_CVPRW_2021_paper.pdf
```bash
python -m ab.nn.select --rank Airia-Det,0.9300,17.51 Sidiki,0.7800,1.74 ByteScene,0.9500,4.44 MobileNet-V2,0.9417,16.38 PyImageSearch,0.8967,45.88 ALONG,0.9467,64.45 neptuneai,0.8367,4.17 EVAI,0.9300,3.35 DataArt,0.9150,54.13 Team-Horizon,0.9233,7.7
```
