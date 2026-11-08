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

Use examples:

```bash
python -m ab.nn.select --rank --model AntYSP,0.998990747,4.33 --model AntSR,0.998995384,4.52 --model z6,0.998983751,5.49 --model koreatechV2,0.998758348,4.35 --model AIO_MAI,0.998950458,41.1 --model IN2GM,0.998964858,46.5 --model LoongSR,0.998943182,5729 --model Godzilla,0.999085887,215160
```

```bash
python -m ab.nn.select --rank --model ZTE_cv,0.999189039,110 --model AntYSP,0.999146900,105 --model Sunrise2046,0.999131040,594 --model OrbSR,0.999160540,1666 --model BVIISR,0.998564511,2106 --model LoongSR,0.999029490,26750 --model Sony_PolyU_LUT,0.998641687,12263
 ```