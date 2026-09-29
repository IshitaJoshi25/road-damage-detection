"""Streamlit UI: upload a road photo and classify its damage with the MobileNetV2 model.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.common import ConfigError, get_class_names, load_config, pretty_class_name, resolve_path
from src.inference import CLASS_COLORS_BGR, FALLBACK_COLOR_BGR

TITLE = "Road Damage Detection and Severity Assessment"
SUBTITLE = "Upload a road photo to identify the type of damage and estimate how severe it is."
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]
DEFAULT_MIN_CONFIDENCE = 0.6
BANNER_HEIGHT_FRACTION = 0.09
BANNER_MIN_HEIGHT_PX = 32
SEVERITY_NOTE = {
    "heuristic": "Severity is a rule-based estimate from the amount of dark, damaged surface in the photo. "
                 "RDD2022 has no Low/Medium/High labels, so it is not a learned prediction.",
    "model": "Severity comes from a classifier trained on human-labeled defect crops.",
    None: "Severity is not available.",
}
CLASS_DESCRIPTIONS = {
    "longitudinal_crack": "Crack running along the direction of traffic.",
    "transverse_crack": "Crack running across the road, perpendicular to traffic.",
    "alligator_crack": "Interconnected cracks forming a pattern like crocodile skin.",
    "pothole": "Bowl-shaped hole where the road surface has broken away.",
}
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 12px; padding: 0.8rem 1rem;
}
.class-dot {display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%; margin-right: 0.45rem;}
.class-row {margin-bottom: 0.35rem;}
</style>
"""


@st.cache_resource(show_spinner="Loading model...")
def get_classifier(model_path: str, model_mtime: float, class_names: tuple[str, ...], image_size: int,
                   preprocessing: str):
    from src.classifier import DamageClassifier

    return DamageClassifier(model_path, list(class_names), image_size, preprocessing)


@st.cache_resource(show_spinner="Loading severity model...")
def get_severity_estimator(model_mtime: float | None, code_mtime: float):
    """Cached on file modification times, so a new severity model or changed code is picked up."""
    import src.severity

    return importlib.reload(src.severity).create_severity_estimator(load_config(), device="cpu")


def severity_estimator(cfg: dict):
    model_path = resolve_path(cfg["severity"].get("model_path"))
    mtime = model_path.stat().st_mtime if model_path and model_path.is_file() else None
    code_mtime = (Path(__file__).parent / "src" / "severity.py").stat().st_mtime
    return get_severity_estimator(mtime, code_mtime)


def hex_color(class_name: str, cfg: dict) -> str:
    """Class colour as #rrggbb (same colours as the rest of the project)."""
    ids = {name: cid for cid, name in get_class_names(cfg).items()}
    b, g, r = CLASS_COLORS_BGR.get(ids.get(class_name, -1), FALLBACK_COLOR_BGR)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_banner(bgr: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Image with a coloured result banner across the top."""
    canvas = bgr.copy()
    h, w = canvas.shape[:2]
    band = max(BANNER_MIN_HEIGHT_PX, int(h * BANNER_HEIGHT_FRACTION))
    cv2.rectangle(canvas, (0, 0), (w, band), color, cv2.FILLED)
    scale = band / 45
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if tw > w - 20:  # shrink long text to fit narrow images
        scale *= (w - 20) / tw
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (11, 11, 11) if luminance > 150 else (255, 255, 255)
    cv2.putText(canvas, text, (10, (band + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness,
                cv2.LINE_AA)
    return canvas


def classify(cfg: dict, model_path: Path, image: Image.Image) -> tuple[np.ndarray, list[dict], dict]:
    """Run the classifier and the severity estimate; returns (bgr, ranked classes, severity result)."""
    ccfg = cfg["classifier"]
    model = get_classifier(str(model_path), model_path.stat().st_mtime, tuple(ccfg["class_names"]),
                           int(ccfg["image_size"]), str(ccfg.get("preprocessing", "mobilenet_v2")))
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    ranked = model.predict(image)
    severity = {"severity": None, "confidence": None, "method": None}
    estimator = severity_estimator(cfg)
    if estimator.available:
        severity = estimator.predict(bgr, detection={"class_name": ranked[0]["class_name"], "whole_image": True},
                                     image_shape=bgr.shape[:2])
    return bgr, ranked, severity


def sidebar(cfg: dict) -> Path:
    ccfg = cfg["classifier"]
    st.sidebar.header("About the model")
    st.sidebar.markdown(f"**MobileNetV2** image classifier · input {ccfg['image_size']}×{ccfg['image_size']} px")
    st.sidebar.caption("It looks at the whole photo and reports the main type of damage. "
                       "It does not draw boxes around individual defects.")
    st.sidebar.subheader("Damage types")
    for name in sorted(ccfg["class_names"], key=pretty_class_name):
        st.sidebar.markdown(
            f"<div class='class-row'><span class='class-dot' style='background:{hex_color(name, cfg)}'></span>"
            f"<b>{pretty_class_name(name)}</b><br><small>{CLASS_DESCRIPTIONS.get(name, '')}</small></div>",
            unsafe_allow_html=True)
    with st.sidebar.expander("Advanced"):
        path = st.text_input("Model file", value=str(resolve_path(ccfg["model_path"])))
    return Path(path)


def show_result(cfg: dict, bgr: np.ndarray, ranked: list[dict], severity: dict, col, file_stem: str) -> None:
    top = ranked[0]
    name, confidence = pretty_class_name(top["class_name"]), top["confidence"]
    level = severity.get("severity")
    b, g, r = (int(hex_color(top["class_name"], cfg)[i:i + 2], 16) for i in (5, 3, 1))
    label = f"{name} {confidence * 100:.0f}%" + (f" | Severity: {level}" if level else "")
    annotated = draw_banner(bgr, label, (b, g, r))

    col.subheader("Result")
    col.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
    ok, png = cv2.imencode(".png", annotated)
    if ok:
        col.download_button("Download result image", png.tobytes(), file_name=f"{file_stem}_result.png",
                            mime="image/png", width="stretch")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Defect type", name)
    c2.metric("Confidence", f"{confidence * 100:.1f}%")
    sev_label = "Severity (estimate)" if severity.get("method") == "heuristic" else "Severity"
    c3.metric(sev_label, level or "Not available")

    min_conf = float(cfg["classifier"].get("min_confidence", DEFAULT_MIN_CONFIDENCE))
    if confidence < min_conf:
        runner_up = pretty_class_name(ranked[1]["class_name"])
        st.warning(f"The model is not sure about this photo ({confidence * 100:.0f}% confidence). "
                   f"It could also be **{runner_up}** ({ranked[1]['confidence'] * 100:.0f}%).")

    st.subheader("Probability for each damage type")
    for item in ranked:
        st.markdown(
            f"<span class='class-dot' style='background:{hex_color(item['class_name'], cfg)}'></span>"
            f"{pretty_class_name(item['class_name'])} — **{item['confidence'] * 100:.1f}%**",
            unsafe_allow_html=True)
        st.progress(min(max(item["confidence"], 0.0), 1.0))
    st.caption(SEVERITY_NOTE.get(severity.get("method"), SEVERITY_NOTE[None]))


def main() -> None:
    st.set_page_config(page_title="Road Damage Assessment", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(TITLE)
    st.caption(SUBTITLE)

    try:
        cfg = load_config()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
    if "classifier" not in cfg:
        st.error("The 'classifier' section is missing from config.yaml.")
        st.stop()
    model_path = sidebar(cfg)

    uploaded = st.file_uploader("Upload a road image", type=UPLOAD_TYPES, help="JPG, JPEG, PNG or WEBP")
    if uploaded is None:
        st.info("Upload a photo of a road surface to get started.")
        return
    try:
        image = Image.open(uploaded).convert("RGB")
    except OSError:
        st.error("Could not read the uploaded file as an image.")
        return

    col_left, col_right = st.columns(2, gap="large")
    col_left.subheader("Uploaded image")
    col_left.image(image, width="stretch")

    if not col_left.button("Detect Damage", type="primary", width="stretch"):
        col_right.subheader("Result")
        col_right.caption("Click **Detect Damage** to analyse the photo.")
        return
    if not model_path.is_file():
        st.error(f"Model file not found: {model_path}. Check the path under Advanced in the sidebar.")
        return

    with st.spinner("Analysing road damage..."):
        bgr, ranked, severity = classify(cfg, model_path, image)
    show_result(cfg, bgr, ranked, severity, col_right, Path(uploaded.name).stem)


if __name__ == "__main__":
    main()
