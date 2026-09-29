"""Discovery of images / Pascal VOC XML files and XML parsing.

The RDD2022 download layout varies between releases (e.g. ``India/train/images``
and ``India/train/annotations/xmls``), so nothing here assumes a fixed folder
structure: files are found recursively and images are matched to XML files by
filename stem.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VocObject:
    """One annotated object from a VOC XML file (pixel coordinates)."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    """Parsed content of one VOC XML file."""

    xml_path: Path
    filename: str | None
    width: int | None
    height: int | None
    objects: list[VocObject] = field(default_factory=list)


@dataclass
class ImageXmlPair:
    """An image matched with its XML annotation, plus a dataset-unique id."""

    uid: str
    image_path: Path
    xml_path: Path


@dataclass
class DiscoveryResult:
    """Everything found while scanning the raw dataset directory."""

    images: list[Path]
    xmls: list[Path]
    pairs: list[ImageXmlPair]
    unmatched_images: list[Path]
    unmatched_xmls: list[Path]


class VocParseError(ValueError):
    """Raised when an XML file cannot be parsed as a VOC annotation."""


def find_files(root: Path, extensions: set[str]) -> list[Path]:
    """Recursively find files under ``root`` whose suffix is in ``extensions``."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def _common_prefix_len(a: Path, b: Path) -> int:
    count = 0
    for x, y in zip(a.parts, b.parts):
        if x != y:
            break
        count += 1
    return count


def _make_uid(image_path: Path, root: Path, duplicated: bool, prefix: str = "") -> str:
    """Stem for unique names; prefix with the relative folder if the stem is duplicated.

    With a ``prefix`` (used for extra datasets) the uid is always prefix + relative path,
    so extra images can never collide with RDD2022 file names.
    """
    if not duplicated and not prefix:
        return image_path.stem
    rel = image_path.relative_to(root).with_suffix("")
    return prefix + re.sub(r"[^A-Za-z0-9_.-]+", "__", "__".join(rel.parts))


def match_images_to_xmls(root: Path, images: list[Path], xmls: list[Path], uid_prefix: str = "") -> DiscoveryResult:
    """Pair images and XML files by filename stem.

    * An XML named after the full image file name (``images.jpeg.xml``) wins; this is
      how two images with the same stem but different extensions are told apart.
    * If the image's own folder contains XML files (XML-next-to-image layout), only
      XML files from that folder are considered - never a same-named file elsewhere.
    * Otherwise (e.g. RDD2022: images/ + annotations/xmls/) the XML closest in the
      folder tree (longest common path prefix) is chosen.
    """
    xml_by_stem: dict[str, list[Path]] = defaultdict(list)
    dirs_with_xml: set[Path] = set()
    for x in xmls:
        xml_by_stem[x.stem].append(x)
        dirs_with_xml.add(x.parent)
    image_stem_counts: dict[str, int] = defaultdict(int)
    for img in images:
        image_stem_counts[img.stem] += 1

    pairs: list[ImageXmlPair] = []
    unmatched_images: list[Path] = []
    used_xmls: set[Path] = set()
    for img in images:
        candidates = [x for x in xml_by_stem.get(img.name, []) if x not in used_xmls and x.parent == img.parent]
        if not candidates:
            candidates = [x for x in xml_by_stem.get(img.stem, []) if x not in used_xmls]
            if img.parent in dirs_with_xml:
                candidates = [x for x in candidates if x.parent == img.parent]
        if not candidates:
            unmatched_images.append(img)
            continue
        xml = max(candidates, key=lambda x: _common_prefix_len(x.parent, img.parent))
        used_xmls.add(xml)
        uid = _make_uid(img, root, image_stem_counts[img.stem] > 1, uid_prefix)
        pairs.append(ImageXmlPair(uid=uid, image_path=img, xml_path=xml))

    unmatched_xmls = [x for x in xmls if x not in used_xmls]
    return DiscoveryResult(images, xmls, pairs, unmatched_images, unmatched_xmls)


def discover_dataset(root: Path, image_extensions: set[str], annotation_extension: str = ".xml",
                     uid_prefix: str = "") -> DiscoveryResult:
    """Scan ``root`` recursively and match images to XML annotations."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {root}\n"
            "Download RDD2022 India and extract it there (or change dataset.raw_dir in config.yaml)."
        )
    images = find_files(root, image_extensions)
    xmls = find_files(root, {annotation_extension.lower()})
    return match_images_to_xmls(root, images, xmls, uid_prefix)


def _find_float(node: ET.Element, tag: str) -> float:
    text = node.findtext(tag)
    if text is None or text.strip() == "":
        raise VocParseError(f"missing <{tag}>")
    try:
        return float(text.strip())
    except ValueError as exc:
        raise VocParseError(f"non-numeric <{tag}>: {text!r}") from exc


def _parse_size(root: ET.Element) -> tuple[int | None, int | None]:
    size = root.find("size")
    if size is None:
        return None, None
    try:
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))
    except ValueError:
        return None, None
    return (width or None), (height or None)


def parse_voc_xml(xml_path: Path) -> tuple[VocAnnotation, list[str]]:
    """Parse a VOC XML file.

    Returns the annotation and a list of per-object problems (objects that could
    not be read are skipped but reported). Raises VocParseError if the file as a
    whole is unreadable.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise VocParseError(f"malformed XML: {exc}") from exc

    width, height = _parse_size(root)
    annotation = VocAnnotation(xml_path=xml_path, filename=root.findtext("filename"), width=width, height=height)
    problems: list[str] = []
    for idx, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            problems.append(f"object {idx}: missing <name> or <bndbox>")
            continue
        try:
            coords = [_find_float(box, t) for t in ("xmin", "ymin", "xmax", "ymax")]
        except VocParseError as exc:
            problems.append(f"object {idx} ({name}): {exc}")
            continue
        annotation.objects.append(VocObject(name, *coords))
    return annotation, problems


EXIF_ORIENTATION_TAG = 274
EXIF_ROTATED_90 = {5, 6, 7, 8}  # orientations that swap width and height


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Return the displayed (width, height) of an image without decoding all pixels.

    EXIF rotation is taken into account, because OpenCV/YOLO load images upright.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        if img.getexif().get(EXIF_ORIENTATION_TAG, 1) in EXIF_ROTATED_90:
            return height, width
        return width, height
