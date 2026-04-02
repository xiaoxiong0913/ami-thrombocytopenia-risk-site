# AMI Thrombocytopenia Risk Site

Public-facing AMI thrombocytopenia risk calculator and reporting bundle.

## Runtime

- Framework: Flask
- Public APIs:
  - `GET /api/config`
  - `GET /healthz`
  - `POST /api/predict`
- Locked runtime model: XGBoost post-calibrated deployment bundle

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

## Render

This repository is prepared for a Render Python Web Service using the root-level `render.yaml`, `Procfile`, and `requirements.txt`.
