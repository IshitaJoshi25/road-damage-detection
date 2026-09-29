"""Streamlit tool for labeling defect crops as Low / Medium / High severity.

Run with:
    streamlit run label_severity.py

Reads data/severity_crops/metadata.csv (from src/crop_defects.py --save-crops)
and saves labels to severity.training.labels_file (default data/severity_crops/labels.csv).
Labeling rules: docs/severity_rubric.md.
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import pandas as pd
import streamlit as st

from src.common import PROJECT_ROOT, ConfigError, load_config, pretty_class_name, resolve_path
from src.crop_defects import METADATA_NAME
from src.inference import draw_predictions
from src.severity import SEVERITY_LEVELS

TITLE = "Severity labeling"
UNCLEAR = "Unclear"
CHOICES = (*SEVERITY_LEVELS, UNCLEAR)
LABEL_COLUMNS = ["crop_path", "class_name", "severity", "labeled_at"]
RUBRIC_PATH = PROJECT_ROOT / "docs" / "severity_rubric.md"
CROP_DISPLAY_WIDTH = 360


def load_labels(path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame(columns=LABEL_COLUMNS)


def save_label(labels_path, crop_path: str, class_name: str, level: str) -> None:
    labels = load_labels(labels_path)
    labels = labels[labels["crop_path"] != crop_path]
    row = {"crop_path": crop_path, "class_name": class_name, "severity": level,
           "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path, index=False)


def on_label(labels_path, item: pd.Series, level: str, advance: bool) -> None:
    save_label(labels_path, item["crop_path"], item["class_name"], level)
    st.session_state.history.append(item["crop_path"])
    if advance:
        st.session_state.pos += 1


def on_undo(labels_path) -> None:
    if st.session_state.history:
        crop_path = st.session_state.history.pop()
        labels = load_labels(labels_path)
        labels[labels["crop_path"] != crop_path].to_csv(labels_path, index=False)


def context_image(item: pd.Series):
    """Full image with the defect box drawn, or None if the image is unavailable."""
    image_path = resolve_path(item.get("image_path")) if isinstance(item.get("image_path"), str) else None
    image = cv2.imread(str(image_path)) if image_path and image_path.is_file() else None
    if image is None:
        return None
    det = {"class_id": int(item["class_id"]), "class_name": item["class_name"], "confidence": 1.0,
           "bbox": [int(item[k]) for k in ("x1", "y1", "x2", "y2")]}
    return cv2.cvtColor(draw_predictions(image, [det], box_thickness=3, show_confidence=False), cv2.COLOR_BGR2RGB)


def sidebar(metadata: pd.DataFrame, labels: pd.DataFrame) -> tuple[list[str], str]:
    st.sidebar.header("Filter")
    classes = sorted(metadata["class_name"].unique())
    chosen = st.sidebar.multiselect("Defect classes", classes, default=classes, format_func=pretty_class_name)
    mode = st.sidebar.radio("Show", ["Unlabeled only", "All crops"], horizontal=True)

    st.sidebar.header("Progress")
    st.sidebar.write(f"{len(labels)} of {len(metadata)} crops labeled")
    if not labels.empty:
        table = pd.crosstab(labels["class_name"].map(pretty_class_name), labels["severity"])
        st.sidebar.dataframe(table.reindex(columns=[c for c in CHOICES if c in table.columns]))
    if RUBRIC_PATH.is_file():
        with st.sidebar.expander("Labeling rubric", expanded=False):
            st.markdown(RUBRIC_PATH.read_text(encoding="utf-8"))
    return chosen, mode


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    try:
        cfg = load_config()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()

    metadata_path = resolve_path(cfg["severity"]["crops_dir"]) / METADATA_NAME
    labels_path = resolve_path(cfg["severity"]["training"]["labels_file"])
    if not metadata_path.is_file():
        st.warning(f"No crops yet ({metadata_path} not found). Create them first, e.g.:")
        st.code("python src/crop_defects.py --source data/yolo/images/train --ground-truth "
                "--max-per-class 150 --save-crops")
        st.stop()

    st.session_state.setdefault("pos", 0)
    st.session_state.setdefault("history", [])
    metadata = pd.read_csv(metadata_path)
    labels = load_labels(labels_path)
    chosen, mode = sidebar(metadata, labels)

    items = metadata[metadata["class_name"].isin(chosen)].reset_index(drop=True)
    label_of = dict(zip(labels["crop_path"], labels["severity"]))
    unlabeled_mode = mode == "Unlabeled only"
    if unlabeled_mode:
        items = items[~items["crop_path"].isin(label_of)].reset_index(drop=True)
        st.session_state.pos = 0
    if items.empty:
        st.success("Nothing left to label in this selection.")
        st.button("Undo last label", on_click=on_undo, args=(labels_path,),
                  disabled=not st.session_state.history)
        return
    st.session_state.pos = min(max(st.session_state.pos, 0), len(items) - 1)
    item = items.iloc[st.session_state.pos]

    st.caption(f"Crop {st.session_state.pos + 1} of {len(items)} · {item['crop_path']}")
    st.subheader(pretty_class_name(item["class_name"]))
    current = label_of.get(item["crop_path"])
    if current:
        st.info(f"Current label: **{current}**")

    cols = st.columns(len(CHOICES))
    for col, level in zip(cols, CHOICES):
        col.button(level, key=f"btn_{level}", width="stretch", type="primary" if level != UNCLEAR else "secondary",
                   on_click=on_label, args=(labels_path, item, level, not unlabeled_mode))

    left, right = st.columns([1, 2])
    crop_path = resolve_path(item["crop_path"])
    if crop_path and crop_path.is_file():
        left.image(str(crop_path), caption="Defect crop", width=CROP_DISPLAY_WIDTH)
    else:
        left.error(f"Crop file missing: {crop_path}")
    context = context_image(item)
    if context is not None:
        right.image(context, caption="Full image (judge size against the lane)", width="stretch")

    nav = st.columns(3)
    nav[0].button("◀ Previous", disabled=unlabeled_mode or st.session_state.pos == 0,
                  on_click=lambda: st.session_state.update(pos=st.session_state.pos - 1))
    nav[1].button("Next ▶", disabled=unlabeled_mode or st.session_state.pos >= len(items) - 1,
                  on_click=lambda: st.session_state.update(pos=st.session_state.pos + 1))
    nav[2].button("Undo last label", on_click=on_undo, args=(labels_path,), disabled=not st.session_state.history)


if __name__ == "__main__":
    main()
