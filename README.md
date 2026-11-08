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

Usage Examples:

Provides the same rank as reported in https://arxiv.org/pdf/2611.00963
```bash
python -m ab.nn.select --rank koreatechV2,0.998758348,4.35 Godzilla,0.999085887,215160 AntSR,0.998995384,4.52 LoongSR,0.998943182,5729 IN2GM,0.998964858,46.5 z6,0.998983751,5.49 AIO_MAI,0.998950458,41.1 AntYSP,0.998990747,4.33
```
