from __future__ import annotations

import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request


APP_DIR = Path(__file__).resolve().parent
CARD_PATH = APP_DIR / "clinical_web_model_card.json"
HTML_PATH = APP_DIR / "clinical_risk_comparison.html"


def _bundle_path(relative_path: str) -> Path:
    return APP_DIR.joinpath(*PureWindowsPath(relative_path).parts)


def _dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "todense"):
        return matrix.todense()
    return matrix


def _coerce_runtime_value(value: Any, feature_spec: dict[str, Any]) -> Any:
    if value in (None, "", "null"):
        return feature_spec["default"]
    if feature_spec.get("key") == "gender":
        mapped = {"M": "Male", "F": "Female"}.get(str(value).strip().upper())
        if mapped:
            return mapped
    if feature_spec["type"] == "select":
        option_values = [option["value"] for option in feature_spec.get("options", [])]
        if any(item in [0, 1, "0", "1"] for item in option_values):
            return float(value)
        return str(value)
    return float(value)


with CARD_PATH.open("r", encoding="utf-8") as fh:
    PAYLOAD = json.load(fh)

RAW_FEATURE_ORDER = list(PAYLOAD["prediction_model"]["raw_feature_order"])
SELECTED_TRANSFORMED_FEATURES = list(PAYLOAD["prediction_model"]["selected_transformed_features"])
FEATURE_SPECS = {item["key"]: item for item in PAYLOAD["prediction_model"]["input_features"]}
MODEL = None
PREPROCESSOR = None
CALIBRATOR = None
TRANSFORMED_INDEX_MAP: dict[str, int] | None = None

app = Flask(__name__)


def _threshold() -> float | None:
    value = PAYLOAD["prediction_model"].get("deployment_threshold")
    if value in (None, ""):
        return None
    return float(value)


def _risk_labels() -> tuple[str, str]:
    labels = PAYLOAD["prediction_model"].get("risk_labels", {})
    return (
        str(labels.get("low", "Lower risk")),
        str(labels.get("high", "Higher risk")),
    )


def _interpretation_templates() -> tuple[str, str]:
    templates = PAYLOAD["prediction_model"].get("interpretation_templates", {})
    return (
        str(templates.get("low", "Predicted risk is below the locked deployment threshold.")),
        str(templates.get("high", "Predicted risk is above the locked deployment threshold.")),
    )


def _ensure_runtime_loaded() -> None:
    global MODEL, PREPROCESSOR, CALIBRATOR, TRANSFORMED_INDEX_MAP
    if MODEL is not None and PREPROCESSOR is not None and TRANSFORMED_INDEX_MAP is not None:
        return

    runtime_payload = PAYLOAD.get("runtime", {})
    bundle_file = runtime_payload.get("bundle_file")
    if bundle_file:
        runtime_bundle = joblib.load(_bundle_path(bundle_file))
        PREPROCESSOR = runtime_bundle["preprocessor"]
        MODEL = runtime_bundle["model"]
        CALIBRATOR = runtime_bundle.get("calibrator")
    else:
        PREPROCESSOR = joblib.load(_bundle_path(runtime_payload["preprocessor_file"]))
        MODEL = joblib.load(_bundle_path(runtime_payload["model_file"]))
        CALIBRATOR = None

    if hasattr(PREPROCESSOR, "get_feature_names_out"):
        transformed_names = [str(item) for item in PREPROCESSOR.get_feature_names_out()]
    else:
        transformed_names = list(RAW_FEATURE_ORDER)
    TRANSFORMED_INDEX_MAP = {name: index for index, name in enumerate(transformed_names)}


def _prediction_frame(inputs: dict[str, Any]) -> pd.DataFrame:
    row = {feature: _coerce_runtime_value(inputs.get(feature), FEATURE_SPECS[feature]) for feature in RAW_FEATURE_ORDER}
    return pd.DataFrame([row], columns=RAW_FEATURE_ORDER)


def _predict(inputs: dict[str, Any]) -> dict[str, Any]:
    _ensure_runtime_loaded()
    frame = _prediction_frame(inputs)
    transformed = _dense(PREPROCESSOR.transform(frame))
    selected_idx = [TRANSFORMED_INDEX_MAP[name] for name in SELECTED_TRANSFORMED_FEATURES if name in TRANSFORMED_INDEX_MAP]
    model_input = transformed[:, selected_idx] if selected_idx and len(selected_idx) == len(SELECTED_TRANSFORMED_FEATURES) else transformed
    raw_probability = float(MODEL.predict_proba(model_input)[:, 1][0])
    risk = raw_probability
    if CALIBRATOR is not None:
        calibrated = CALIBRATOR.predict_proba(np.asarray(raw_probability).reshape(-1, 1))
        risk = float(calibrated[:, 1][0])

    threshold = _threshold()
    low_label, high_label = _risk_labels()
    low_text, high_text = _interpretation_templates()
    if threshold is not None and risk >= threshold:
        risk_group = high_label
        interpretation = high_text
    elif threshold is not None:
        risk_group = low_label
        interpretation = low_text
    else:
        risk_group = ""
        interpretation = ""

    details = []
    if interpretation:
        details.append(interpretation)
    if threshold is not None:
        details.append(f"Locked deployment threshold: {threshold:.4f}")
    details.append(f"Model alignment: {PAYLOAD['prediction_model']['deployment_note']}")
    details.append(f"Input scope: {PAYLOAD['metadata']['prediction_model_feature_count']} harmonized ICU variables")
    details.append(f"External validation AUROC: {PAYLOAD['prediction_model']['performance']['external_auroc']:.3f}")

    top_drivers = [str(item.get("label", "")).strip() for item in PAYLOAD["prediction_model"].get("top_drivers", [])[:3]]
    top_drivers = [item for item in top_drivers if item]
    if top_drivers:
        details.append(f"Leading drivers: {', '.join(top_drivers)}")

    return {
        "prediction_model": {
            "title": f"{PAYLOAD['prediction_model']['name']} deployment model",
            "risk": risk,
            "risk_percent": risk * 100.0,
            "threshold": threshold,
            "risk_group": risk_group,
            "interpretation": interpretation,
            "details": details,
        }
    }


@app.get("/")
def index():
    return Response(HTML_PATH.read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/api/config")
def config():
    payload = dict(PAYLOAD)
    payload.pop("runtime", None)
    return jsonify(payload)


@app.post("/api/predict")
def predict():
    request_payload = request.get_json(silent=True) or {}
    try:
        return jsonify(_predict(request_payload.get("inputs", {})))
    except Exception as exc:
        app.logger.exception("Prediction request failed.")
        return jsonify({"error": "prediction_failed", "detail": str(exc)}), 500


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "model": PAYLOAD["prediction_model"]["name"], "runtime_loaded": MODEL is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8765")))
