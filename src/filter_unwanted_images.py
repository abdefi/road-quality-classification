import os
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.util import img_as_float

input_folder = "../images/all_images"
nonstreet_folder = "../images/maybe_nonstreet"
os.makedirs(nonstreet_folder, exist_ok=True)

SAT_THRESHOLD = 50
VAL_MIN = 30
VAL_MAX = 220
CROP_SIZE = 100

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_folder, filename)
    img = io.imread(path)

    if img is None or img.ndim != 3 or img.shape[2] < 3:
        continue

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = CROP_SIZE // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    crop = img[y1:y2, x1:x2, :3]

    img_float = img_as_float(crop)
    hsv = color.rgb2hsv(img_float)

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    mean_saturation = np.mean(saturation) * 255
    mean_brightness = np.mean(brightness) * 255
    max_brightness = np.max(brightness) * 255

    if (mean_saturation >= SAT_THRESHOLD or
        mean_brightness <= VAL_MIN or
        mean_brightness >= VAL_MAX or
        max_brightness < 10):
        dest_path = os.path.join(nonstreet_folder, filename)
        io.imsave(dest_path, img_as_ubyte(img))
        os.remove(path)
