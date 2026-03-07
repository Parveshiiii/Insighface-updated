# InsightFace — Ultra-Parallel GPU Batching Fork

> **Note: This README was written by AI.**

This is a fork of the [original InsightFace](https://github.com/deepinsight/insightface) Python package, modified to support **ultra-parallel GPU batching** for significantly faster face embedding generation.

---

## What's Changed

| File | Change |
|---|---|
| `model_zoo/arcface_onnx.py` | Added `get_batch(img, faces)` method — processes N faces in a **single ONNX Runtime GPU call** instead of N sequential calls |
| `app/face_analysis.py` | Updated `FaceAnalysis.get()` to automatically use `get_batch()` when available |
| `utils/face_align.py` | Fixed deprecated `SimilarityTransform.estimate()` → `SimilarityTransform.from_estimate()` |

### Performance Impact
- **Before**: Each face embedding computed one at a time (1 GPU call per face)
- **After**: All faces in an image batched into a single GPU call (N faces = 1 call)
- Fully compatible with `onnxruntime-gpu` CUDA parallel execution

---

## Install

### From GitHub (recommended)

```bash
pip install git+https://github.com/Parveshiiii/Insighface-updated.git
```

### Pin to a specific commit (for reproducible environments)

```bash
pip install git+https://github.com/Parveshiiii/Insighface-updated.git@34b152e
```

### Add to requirements.txt

```
git+https://github.com/Parveshiiii/Insighface-updated.git
```

### Prerequisites

You must install an ONNX Runtime backend separately:

```bash
# For GPU (CUDA) — recommended for production
pip install onnxruntime-gpu

# For CPU only
pip install onnxruntime
```

---

## Quick Usage

```python
import cv2
from insightface.app import FaceAnalysis

# Initialize — automatically uses get_batch() for parallel GPU embedding
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread("your_image.jpg")
faces = app.get(img)  # All face embeddings computed in ONE GPU batch!

for face in faces:
    print(face.embedding.shape)  # (512,) — L2 normalized embedding
```

### Advanced: Cross-Image Batching (Maximum GPU Throughput)

```python
from insightface.utils import face_align

rec_model = app.models['recognition']

# Phase 1: Detect & align faces from multiple images
all_crops = []
for img in list_of_imgs:
    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
    for i in range(bboxes.shape[0]):
        aimg = face_align.norm_crop(img, landmark=kpss[i], image_size=112)
        all_crops.append(aimg)

# Phase 2: One batched GPU call for ALL faces across ALL images
BATCH_SIZE = 64
for i in range(0, len(all_crops), BATCH_SIZE):
    chunk = all_crops[i : i + BATCH_SIZE]
    embeddings = rec_model.get_feat(chunk)  # Shape: (BATCH_SIZE, 512)
```

---

## License

The code is released under the **MIT License**.

> The pretrained models provided with this library are available for **non-commercial research purposes only**.

---

*This README was written by AI.*
