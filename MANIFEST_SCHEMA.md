# Manifest schema

The plugin polls this URL once a day (configurable via `VOV_MODEL_REGISTRY` constant):

```
https://models.valetops.com/vision/manifest.json
```

## Schema

```json
{
  "schema_version": 1,
  "vehicle_classifier": {
    "version": "3.2.0",
    "released": "2026-09-14T00:00:00Z",
    "size_mb": 18.4,
    "classes": 2847,
    "url": "https://models.valetops.com/vision/vehicle-classifier-v3.2.0.onnx",
    "sha256": "abc123def456...",
    "min_plugin_version": "1.2.0",
    "input_shape": [1, 3, 224, 224],
    "label_map_url": "https://models.valetops.com/vision/vehicle-classifier-v3.2.0.labels.json",
    "color_head_url": "https://models.valetops.com/vision/vehicle-color-v1.4.0.onnx",
    "changelog": [
      "+47 model-year 2026 variants",
      "+2027 Hyundai Palisade XRT",
      "+2027 Tesla Model Y refresh",
      "+2026 Lucid Air Sapphire",
      "improved dark-truck accuracy 94.1% → 96.8%",
      "fixed Subaru Outback / Forester confusion"
    ],
    "test_accuracy": {
      "top1": 0.961,
      "top5": 0.994,
      "by_year": { "2024": 0.972, "2025": 0.964, "2026": 0.951, "2027": 0.918 }
    }
  },
  "plate_detector": {
    "version": "2024.10.0",
    "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx",
    "size_mb": 7.4,
    "sha256": "...",
    "input_size": 384
  },
  "plate_ocr": {
    "version": "2024.10.0",
    "url": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global.onnx",
    "size_mb": 5.0,
    "sha256": "...",
    "charset": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
    "max_len": 9
  }
}
```

## Fields

- `version`: SemVer. Plugin compares with installed version, pulls if higher.
- `min_plugin_version`: If the customer's plugin is older than this, skip the update (compatibility break).
- `sha256`: Required. Plugin verifies the download before swapping the cached model. Mismatch → reject.
- `label_map_url`: JSON array of class names matching the classifier's output indices. Loaded once with the model.
- `color_head_url`: Optional. If present, a smaller secondary model that runs on the same feature map for color. Trained jointly but published separately so the color model can iterate faster than the make/model model.
- `changelog`: Free-text array. Shown in admin UI when a customer asks "what's new?"
- `test_accuracy.by_year`: Plugin shows a warning if accuracy on the customer's typical model-year drops more than 3% in a release.

## Publishing flow (the pipeline's `publish.py` does this)

1. Train + export ONNX. Run hold-out test set. Compute `test_accuracy`.
2. If top-1 < previous_top1 - 0.5%, **abort** (regression gate).
3. Compute sha256 of the ONNX file.
4. Upload ONNX + label-map JSON to R2.
5. Compose a new manifest with bumped version, new URL, sha256, changelog.
6. Validate JSON against this schema.
7. PUT new manifest to R2 at `manifest.json`.
8. Send Slack notification: "Published v3.2.0, +47 model-year 2026 variants, top-1 96.1%."
9. Plugin polling intervals are unbiased — within 24 hours every active customer site has the new model swapped in for the next scan.
