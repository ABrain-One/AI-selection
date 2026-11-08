# AI Ranking

Ranks neural networks by the **accuracy–resource trade-off** and returns one
score per model. There is no weight to choose: the exchange rate between
accuracy and cost is derived from the candidates' own frontier.

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

## Use as an API

```python
import pandas as pd
from ab.nn.select import select

df = pd.DataFrame({
    "name":     ["B0",   "B1",  "B2",  "B3",  "B4",  "B5",  "B6",  "B7"],
    "accuracy": [0.771, 0.791, 0.801, 0.816, 0.829, 0.836, 0.840, 0.843],
    "flops_B":  [0.39,  0.70,  1.0,   1.8,   4.2,   9.9,   19,    37],
    "params_M": [5.3,   7.8,   9.2,   12,    19,    30,    43,    66],
})

r = select(df, res_col=["flops_B", "params_M"], rank=True)

r["winner"]["name"]       # 'B0'      <- the selected model
r["winner"]["score"]      # 0.7140    <- its score, higher is better
r["verdict"]["level"]     # 'USABLE' | 'UNVERIFIED' | 'UNUSABLE'
r["ranking"]              # DataFrame: name, score, rank, ...
```

### Input

One row per model, in a `pandas.DataFrame`:

| column | required | meaning |
|---|---|---|
| `name` | yes | model identifier |
| `accuracy` | yes | quality in **(0, 1)**, strictly between 0 and 1 |
| one or more resource columns | yes | anything you spend: latency, FLOPs, parameters, memory, energy. Each **> 0** |

`select(df, res_col=...)` names the resource column(s) — a string for one, a
list for several. All models must be measured on the **same test set and the
same device**, and each resource in **one fixed unit**.

Quality that is not already in (0, 1) must be converted first. For PSNR in dB,
`accuracy = 1 - 10 ** (-PSNR / 10)`. For top-1, mAP, mIoU, SSIM or F1, use the
value as it is.

### Output

`select()` returns a dict. The three things most users need:

| key | what it is |
|---|---|
| `winner["name"]` | **the selected model** |
| `winner["score"]` | **its score** — higher is better |
| `ranking` | a `DataFrame`, winner first: `name`, `score`, `rank`, `accuracy`, each resource, `sigma`, `eps_eff`, `margin_to_next`, `rank_low`, `rank_high`, `note` |

Also returned: `verdict`, `status`, `prices` (one price per resource axis),
`rate`, `price_free` (dominations true at every price), `front`, `fit`,
`envelope`, `axis_drop`, `next`, `notes`, `problems`. `select()` prints
nothing and raises nothing; bad input comes back as `status="bad_input"` with
the offending rows listed in `problems`.

### The score

```
score   = 1 / (1 + eps_eff)
eps_eff = eps * product over axes of (resource ** lambda)      eps = 1 - accuracy
```

`eps_eff` is **the error the model effectively costs once its resources are
charged** at the derived price `lambda`; `score` is that mapped into (0, 1) so
that **higher is better**.

Its one useful property: it is computed from a single model's own accuracy and
resources, and from nothing else. **Adding or removing other models cannot
change it** — unlike `rank`, or any score normalised over the candidate set.
So scores are comparable across runs and across published tables.

Two things it is *not* invariant to:

- **the resource unit** — milliseconds vs seconds shifts every score by the
  same constant, so the ordering is untouched but the numbers are not
  comparable unless the unit is fixed;
- **`lambda`** — only compare scores derived at the same price.

### Read the verdict before the ranking

`verdict["level"]` says whether the ranking means anything. A clean run prints
nothing; a problem prints in red, first.

| level | meaning |
|---|---|
| `USABLE` | the error floor is confirmed by a nested F-test, so an exchange rate exists and the ranking is a genuine trade-off ranking |
| `UNVERIFIED` | too few points on the frontier to test the floor, or its knee lies outside the measured range — the ranking is a guess |
| `UNUSABLE` | the floor is rejected: no exchange rate exists in this data, and the ranking collapses to resource order |

`UNVERIFIED` and `UNUSABLE` still return a full ranking — labelled, not
withheld. What holds regardless of the verdict is `price_free`: a model beaten
on accuracy **and** on every resource can never win at any price.

### How many models are needed

Counts refer to candidates that **survive the cull**, not those supplied.

| surviving | behaviour |
|---|---|
| **8 or more** | the floor test identifies a rate reliably — `USABLE` expected |
| 6–7 | the papers' threshold, but noise-sensitive: identification 81% at 1% accuracy noise, 9% at 3% |
| 3–5 | the floor cannot be tested; a ranking is returned, labelled `UNVERIFIED` |

Fewer still ranks usefully: the exponent's precision is flat from three to ten
candidates when a floor is present, and `price_free` dominations hold at any
price. On MAI 2021, three surviving candidates of ten recovered the official
top two and identified six of the nine entries as dominated.

Competition leaderboards rarely reach eight by construction: their rules fix
quality at a threshold and compete on cost, so the surviving frontier is short
and carries little curvature.

## Usage examples

Ranked by the accuracy–time trade-off, no weight to choose. Any bare argument
containing a comma is a candidate: `name,accuracy,p1[,p2,...]`.

1. MAI 2026 INT8, SSIM vs inference time ms (NPU) — [arXiv:2611.00963](https://arxiv.org/pdf/2611.00963)
```bash
python -m ab.nn.select --rank koreatechV2,0.8499,4.35 Godzilla,0.8831,215160 AntSR,0.8731,4.52 LoongSR,0.8586,5729 IN2GM,0.8701,46.5 z6,0.8699,5.49 AIO_MAI,0.8634,41.1 AntYSP,0.8729,4.33
```

2. MAI 2026 INT8, PSNR converted to (0,1) vs 3 resources — NPU ms, CPU ms, model size (KB) — [arXiv:2611.00963](https://arxiv.org/pdf/2611.00963)
```bash
python -m ab.nn.select --rank koreatechV2,0.998758348,4.35,464,77.1 Godzilla,0.999085887,215160,215160,16253 AntSR,0.998995384,4.52,404,65.3 LoongSR,0.998943182,5729,5729,708 IN2GM,0.998964858,46.5,526,64.3 z6,0.998983751,5.49,330,67.4 AIO_MAI,0.998950458,41.1,610,96.1 AntYSP,0.998990747,4.33,338,54.9
```

3. EfficientNet B0–B7, ImageNet top-1 vs FLOPs and params (2 resources simultaneously) — [arXiv:1905.11946](https://arxiv.org/pdf/1905.11946), Table 2
```bash
python -m ab.nn.select --rank B7,0.843000000,37,66 B0,0.771000000,0.39,5.3 B4,0.829000000,4.2,19 B6,0.840000000,19,43 B2,0.801000000,1,9.2 B1,0.791000000,0.7,7.8 B3,0.816000000,1.8,12 B5,0.836000000,9.9,30
```

4. YOLO11, mAP vs 4 resources — T4 TensorRT10 ms, CPU ONNX ms, params (M), FLOPs (B) — [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
```bash
python -m ab.nn.select --rank YOLO11l,0.534000000,6.2,238.6,25.3,87.2 YOLO11m,0.515000000,4.7,183.2,20.1,68.1 YOLO11x,0.547000000,11.3,462.8,56.9,195.3 YOLO11n,0.395000000,1.5,56.1,2.6,6.5 YOLO11s,0.470000000,2.5,90.0,9.4,21.6
```

5. MAI 2021 camera scene detection, Top-1 vs A11 Bionic runtime — [CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Ignatov_Fast_and_Accurate_Quantized_Camera_Scene_Detection_on_Smartphones_Mobile_CVPRW_2021_paper.pdf)
```bash
python -m ab.nn.select --rank Airia-Det,0.9300,17.51 Sidiki,0.7800,1.74 ByteScene,0.9500,4.44 MobileNet-V2,0.9417,16.38 PyImageSearch,0.8967,45.88 ALONG,0.9467,64.45 neptuneai,0.8367,4.17 EVAI,0.9300,3.35 DataArt,0.9150,54.13 Team-Horizon,0.9233,7.7
```

A CSV works too, with headers guessed or named:
```bash
python -m ab.nn.select models.csv --rank --acc-col top1 --res-col latency_ms --res-col params_M
```

### Other options

| flag | effect |
|---|---|
| `--rank` | order every model winner to loser (otherwise only the winner) |
| `--out-csv FILE` | write the ranking to CSV (`-` for stdout) |
| `--alpha 0.05` | coverage level for the rate interval |
| `--boot N` | max bootstrap refits; skipped automatically once the frontier has 50+ points |
| `--no-color`, `--quiet` | plain text / suppress the report |

Exit codes: `0` clean, `1` `UNVERIFIED` or `UNUSABLE`, `2` bad input.

Scale: thousands of candidates are fine — they collapse to a frontier of a few
hundred, and the fit runs on that. 20,000 models rank in about 1 second.