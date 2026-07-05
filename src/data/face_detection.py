"""Optional face detection / cropping.

The OSDFD paper detects faces with dlib, enlarges the box by 1.3x and resizes to
224x224 (Sec. IV-A). Most FF++-style pipelines pre-extract cropped face frames,
so cropping is **disabled by default** here. When enabled, a pluggable backend
crops the largest detected face with the paper's 1.3x margin.

Backends:
  * ``opencv``     -- Haar cascade, always available with opencv.
  * ``retinaface`` -- more accurate, requires ``retina-face`` (optional).
  * ``dlib``       -- the paper's detector, requires ``dlib`` (optional).

If detection fails the original image is returned unchanged.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class FaceCropper:
    """Crop the largest face with a configurable margin.

    Args:
        backend: One of ``"opencv"``, ``"retinaface"``, ``"dlib"``.
        margin: Enlargement factor applied to the detected box (paper: 1.3).
    """

    def __init__(self, backend: str = "opencv", margin: float = 1.3) -> None:
        self.backend = backend
        self.margin = margin
        self._detector = None

    def _lazy_init(self) -> None:
        if self._detector is not None:
            return
        if self.backend == "opencv":
            import cv2

            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(path)
        elif self.backend == "dlib":  # pragma: no cover - optional dependency
            import dlib

            self._detector = dlib.get_frontal_face_detector()
        elif self.backend == "retinaface":  # pragma: no cover - optional dependency
            from retinaface import RetinaFace

            self._detector = RetinaFace
        else:
            raise ValueError(f"Unknown face-detection backend: {self.backend}")

    def _detect_box(self, img: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return the largest face box as ``(x1, y1, x2, y2)`` or ``None``."""
        if self.backend == "opencv":
            import cv2

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self._detector.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            return x, y, x + w, y + h
        if self.backend == "dlib":  # pragma: no cover - optional dependency
            dets = self._detector(img, 1)
            if len(dets) == 0:
                return None
            d = max(dets, key=lambda r: r.width() * r.height())
            return d.left(), d.top(), d.right(), d.bottom()
        # retinaface  # pragma: no cover - optional dependency
        res = self._detector.detect_faces(img)
        if not isinstance(res, dict) or len(res) == 0:
            return None
        face = max(res.values(), key=lambda f: (
            (f["facial_area"][2] - f["facial_area"][0])
            * (f["facial_area"][3] - f["facial_area"][1])
        ))
        x1, y1, x2, y2 = face["facial_area"]
        return x1, y1, x2, y2

    def __call__(self, image: Image.Image) -> Image.Image:
        self._lazy_init()
        img = np.array(image.convert("RGB"))
        box = self._detect_box(img)
        if box is None:
            return image

        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_w = (x2 - x1) * self.margin / 2.0
        half_h = (y2 - y1) * self.margin / 2.0
        nx1 = max(0, int(cx - half_w))
        ny1 = max(0, int(cy - half_h))
        nx2 = min(w, int(cx + half_w))
        ny2 = min(h, int(cy + half_h))
        return image.crop((nx1, ny1, nx2, ny2))
