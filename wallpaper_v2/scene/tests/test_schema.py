"""
P3-T1 self-test: scene-v1.schema.json is valid, and a hand-written minimal
example validates while known-bad examples are rejected.

Not a pytest suite (the repo has none, see harness/test_regions.py) -- a
plain assert-based script. Exits non-zero on any failure.

Usage:
    python3 scene/tests/test_schema.py
"""
import copy
import json
import sys
from pathlib import Path

import jsonschema

SCENE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = SCENE_DIR / "schema" / "scene-v1.schema.json"


def load_schema():
    return json.loads(SCHEMA_PATH.read_text())


MINIMAL_EXAMPLE = {
    "version": 1,
    "pipeline_sha": "58dc47c",
    "image": {"width": 3840, "height": 2560},
    "camera": {"f_px": 1774.3, "cx": 1920.0, "cy": 1280.0},
    "gravity": [-0.024, -0.999, -0.031],
    "planes": [{
        "id": 0,
        "normal": [0.880, -0.004, 0.474],
        "d": 2.97,
        "u_axis": [0.474, 0.0, -0.880],
        "v_axis": [-0.004, -0.999, -0.007],
        "origin": [0.31, 0.02, 3.11],
        "px_share": 0.668,
        "extent_m": [4.97, 3.69],
        "confidence": 0.91,
        "model_vs_depth_deg": 5.0,
        "suggested_label": "right wall",
    }],
    "regions": [{
        "id": 0,
        "plane": 0,
        "npix": 161623,
        "area_m2": 16.3,
        "centroid_px": [2790, 900],
        "bbox_px": [2100, 320, 3800, 1900],
    }],
    "quality": {
        "vertical_extent_m": 3.55,
        "wall_pair_angles_deg": [78, 88],
        "non_manhattan_major_pairs": 0,
        "gravity_tilt_deg": 1.9,
        "gravity_ambiguous": False,
        "wall_coverage": 0.98,
        "irradiance_polarity_corr": -0.965,
        "max_model_vs_depth_deg": 15.0,
    },
    "shading": {"scale": 1.6},
    "pattern": {"default_width_m": 0.53},
}


def region_references_valid(scene):
    """Cross-field check JSON Schema itself cannot express."""
    plane_ids = {p["id"] for p in scene["planes"]}
    return all(r["plane"] in plane_ids for r in scene["regions"])


def validate_scene(scene, schema):
    jsonschema.validate(instance=scene, schema=schema)
    if not region_references_valid(scene):
        raise ValueError("region references a non-existent plane")


def test_schema_is_valid_json_schema():
    schema = load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    print("  ok: schema self-validates as draft 2020-12")


def test_minimal_example_validates():
    schema = load_schema()
    validate_scene(MINIMAL_EXAMPLE, schema)
    print("  ok: hand-written minimal example validates")


def test_known_bad_examples_rejected():
    schema = load_schema()
    cases = {}

    missing_version = copy.deepcopy(MINIMAL_EXAMPLE)
    del missing_version["version"]
    cases["missing version"] = missing_version

    bad_arity_normal = copy.deepcopy(MINIMAL_EXAMPLE)
    bad_arity_normal["planes"][0]["normal"] = [1.0, 0.0]
    cases["normal with wrong number of components"] = bad_arity_normal

    bad_region_ref = copy.deepcopy(MINIMAL_EXAMPLE)
    bad_region_ref["regions"][0]["plane"] = 99
    cases["region.plane points at a non-existent plane"] = bad_region_ref

    non_positive_d = copy.deepcopy(MINIMAL_EXAMPLE)
    non_positive_d["planes"][0]["d"] = -1.0
    cases["d <= 0"] = non_positive_d

    n_ok = 0
    for name, bad in cases.items():
        try:
            validate_scene(bad, schema)
        except (jsonschema.ValidationError, ValueError):
            n_ok += 1
            print(f"  ok: rejected -- {name}")
        else:
            raise AssertionError(f"known-bad example was NOT rejected: {name}")

    assert n_ok == len(cases), f"only {n_ok}/{len(cases)} bad examples rejected"
    return n_ok, len(cases)


def main():
    test_schema_is_valid_json_schema()
    test_minimal_example_validates()
    n_ok, n_total = test_known_bad_examples_rejected()
    print(f"\nall P3-T1 schema tests passed ({n_ok}/{n_total} negative examples rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
