# Severity labeling rubric

Use these rules when labeling crops in `label_severity.py`. They are simplified
from the severity levels in **ASTM D6433** (*Standard Practice for Roads and
Parking Lots Pavement Condition Index Surveys*) so that they can be judged from
a single photo.

Photos have no scale, so estimate sizes against familiar objects. A lane or
edge marking is about 10–15 cm wide, a car tyre is about 20 cm wide, and a
typical lane is about 3.5 m wide.

Label what you see **inside the box**. When you are torn between two levels,
pick the lower one. Use **Unclear** if the crop is too blurry, too dark or
too small to judge, or if it is not actually road damage.

## Longitudinal crack (D00) and transverse crack (D10)

| Level | What it looks like |
|---|---|
| **Low** | Thin, hairline crack, under about 1 cm wide. Clean edges, no crumbling. Can be filled or sealed. |
| **Medium** | Clearly open crack, about 1–7 cm wide, **or** a thinner crack with some small random cracks next to it. Edges may be slightly broken. |
| **High** | Wide crack, over about 7 cm, **or** any crack with heavy crumbling or spalled edges or dense branching cracks around it. Pieces may be missing. |

## Alligator crack (D20)

| Level | What it looks like |
|---|---|
| **Low** | Fine, mostly parallel hairline cracks with few connections. The surface is not broken up. |
| **Medium** | Cracks joined into a clear network or "crocodile skin" pattern. Edges may be lightly crumbled. |
| **High** | A well-developed network with distinct pieces. Edges are spalled, pieces are loose or missing, and the area may be sunken. |

## Pothole (D40)

ASTM D6433 rates potholes by depth and diameter. Depth is hard to see in a
photo, so use shadow, visible edges and the base material as cues.

| Level | What it looks like |
|---|---|
| **Low** | Small (roughly under 45 cm across) and shallow (under about 2.5 cm). The surface layer is worn away, with no deep hole. |
| **Medium** | Medium size (about 20–75 cm) with clear depth, where the bottom is visibly lower than the road and has a shadow. Or small but deep. |
| **High** | Large (over about 45 cm) and deep (over about 5 cm). Base or soil is exposed, water may be standing in it, or it takes up a big part of the lane. |

## Tips for consistent labels

- Label at least **30 crops per level**, and ideally about 100 per defect class
  overall. `train_severity.py` needs at least `min_per_level` (default 5) crops
  for each of Low / Medium / High.
- Look at the full-image view in the tool as well as the crop, to judge size
  against the lane.
- Label in one sitting where possible, and re-check your first 20 labels at the
  end. People's standards drift while labeling.
- If two people label, have both label the same 50 crops first and compare. If
  they disagree a lot, refine these rules before continuing.
