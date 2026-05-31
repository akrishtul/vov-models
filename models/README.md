# Third-party ONNX models — mirrored for CORS

These ONNX files are byte-identical mirrors of upstream releases. We host them
here because the original GitHub release URLs 302-redirect to
`release-assets.githubusercontent.com`, which does NOT send
`Access-Control-Allow-Origin` headers — browsers block the fetch when the
calling origin isn't `github.com`. jsDelivr serves files from a repo tree
with proper CORS headers, so re-publishing here lets the in-browser pipeline
actually load them.

## Files

- **`yolo-v9-t-384-license-plates-end2end.onnx`** (7.5 MB)
  - Upstream: <https://github.com/ankandrew/open-image-models/releases/tag/assets>
  - Author: Ankandrew (Open Image Models)
  - License: MIT
  - Purpose: license-plate localization (input 384×384, end-to-end NMS)

- **`cct_s_v2_global.onnx`** (5.1 MB)
  - Upstream: <https://github.com/ankandrew/fast-plate-ocr/releases/tag/arg-plates>
  - Author: Ankandrew (Fast Plate OCR)
  - License: MIT
  - Purpose: license-plate OCR (input 64×128 grayscale, output 9-char string)

No modifications. Re-upload the source binaries when upstream cuts a new
release; bump `version` in `pipeline/manifest.json` correspondingly.
