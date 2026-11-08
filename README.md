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

Usage examples (ranked by the accuracy–time trade-off, no weight to choose):

1. MAI 2026 INT8, SSIM vs inference time ms (NPU): https://arxiv.org/pdf/2611.00963
```bash
python -m ab.nn.select --rank koreatechV2,0.8499,4.35 Godzilla,0.8831,215160 AntSR,0.8731,4.52 LoongSR,0.8586,5729 IN2GM,0.8701,46.5 z6,0.8699,5.49 AIO_MAI,0.8634,41.1 AntYSP,0.8729,4.33
```

2. MAI 2026 INT8, PSNR converted to (0,1) vs 3 resources — NPU ms, CPU ms, and model size (KB): https://arxiv.org/pdf/2611.00963
```bash
python -m ab.nn.select --rank koreatechV2,0.998758348,4.35,464,77.1 Godzilla,0.999085887,215160,215160,16253 AntSR,0.998995384,4.52,404,65.3 LoongSR,0.998943182,5729,5729,708 IN2GM,0.998964858,46.5,526,64.3 z6,0.998983751,5.49,330,67.4 AIO_MAI,0.998950458,41.1,610,96.1 AntYSP,0.998990747,4.33,338,54.9
```

3. EfficientNet B0–B7, ImageNet top-1 vs FLOPs and params (2 resources simultaneously): https://arxiv.org/pdf/1905.11946, Table 2
```bash
python -m ab.nn.select --rank B7,0.843000000,37,66 B0,0.771000000,0.39,5.3 B4,0.829000000,4.2,19 B6,0.840000000,19,43 B2,0.801000000,1,9.2 B1,0.791000000,0.7,7.8 B3,0.816000000,1.8,12 B5,0.836000000,9.9,30
```

4. YOLO11, mAP vs 4 resources — T4 TensorRT10 ms, CPU ONNX ms, params (M), FLOPs (B): https://docs.ultralytics.com/models/yolo11/
```bash
python -m ab.nn.select --rank YOLO11l,0.534000000,6.2,238.6,25.3,87.2 YOLO11m,0.515000000,4.7,183.2,20.1,68.1 YOLO11x,0.547000000,11.3,462.8,56.9,195.3 YOLO11n,0.395000000,1.5,56.1,2.6,6.5 YOLO11s,0.470000000,2.5,90.0,9.4,21.6
```

6. MAI 2021 camera scene detection, Top-1 vs A11 Bionic runtime: https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Ignatov_Fast_and_Accurate_Quantized_Camera_Scene_Detection_on_Smartphones_Mobile_CVPRW_2021_paper.pdf
```bash
python -m ab.nn.select --rank Airia-Det,0.9300,17.51 Sidiki,0.7800,1.74 ByteScene,0.9500,4.44 MobileNet-V2,0.9417,16.38 PyImageSearch,0.8967,45.88 ALONG,0.9467,64.45 neptuneai,0.8367,4.17 EVAI,0.9300,3.35 DataArt,0.9150,54.13 Team-Horizon,0.9233,7.7
```
