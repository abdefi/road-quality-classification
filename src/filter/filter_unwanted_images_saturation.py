import os
import shutil
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import streamlit as st
from skimage import color, io
from skimage.util import img_as_float, img_as_ubyte

def clear_folder(folder: str) -> None:
    """
    Delete and recreate a folder.

    :param folder: Path to the folder.
    :return:
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def get_center_crop(img: np.ndarray, crop_size: int) -> Optional[np.ndarray]:
    """
    Get a square crop from the center of the image.

    :param img: Input image as numpy array.
    :param crop_size: Size of the square crop.
    :return: Cropped image as numpy array or None if invalid.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2, :3]

def analyze_crop(crop: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate mean saturation, mean brightness, and max brightness for a crop.

    :param crop: Cropped image as numpy array.
    :return: Tuple of (mean_saturation, mean_brightness, max_brightness).
    """
    img_float = img_as_float(crop)
    hsv = color.rgb2hsv(img_float)
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    mean_saturation = np.mean(saturation) * 255
    mean_brightness = np.mean(brightness) * 255
    max_brightness = np.max(brightness) * 255
    return mean_saturation, mean_brightness, max_brightness

def is_non_street_image(
    mean_saturation: float, mean_brightness: float, max_brightness: float,
    sat_thresh: int, v_min: int, v_max: int, max_bright_px_thresh: int
) -> Tuple[bool, str]:
    """
    Decide if an image is 'non-street' and return the reason.

    :param mean_saturation: Mean saturation value.
    :param mean_brightness: Mean brightness value.
    :param max_brightness: Max brightness value.
    :param sat_thresh: Saturation threshold.
    :param v_min: Minimum brightness.
    :param v_max: Maximum brightness.
    :param max_bright_px_thresh: Max brightness pixel threshold.
    :return: Tuple (is_non_street, reason).
    """
    reasons = []
    if mean_saturation >= sat_thresh:
        reasons.append(f"Mean Saturation ({mean_saturation:.1f}) >= {sat_thresh}")
    if mean_brightness <= v_min:
        reasons.append(f"Mean Brightness ({mean_brightness:.1f}) <= {v_min}")
    if mean_brightness >= v_max:
        reasons.append(f"Mean Brightness ({mean_brightness:.1f}) >= {v_max}")
    if max_brightness < max_bright_px_thresh:
        reasons.append(f"Max Brightness ({max_brightness:.1f}) < {max_bright_px_thresh}")
    return len(reasons) > 0, "; ".join(reasons)

def process_images(
    input_fldr: str, good_imgs_fldr: str, non_street_prev_fldr: str,
    sat_thresh: int, v_min: int, v_max: int, max_bright_px_thresh: int, c_size: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Analyze all images in the input folder and sort them into 'good' or 'non-street'.

    :param input_fldr: Input folder path.
    :param good_imgs_fldr: Output folder for good images.
    :param non_street_prev_fldr: Output folder for non-street images.
    :param sat_thresh: Saturation threshold.
    :param v_min: Minimum brightness.
    :param v_max: Maximum brightness.
    :param max_bright_px_thresh: Max brightness pixel threshold.
    :param c_size: Crop size.
    :return: Tuple (list of non-street images with reasons, count of good images).
    """
    clear_folder(good_imgs_fldr)
    clear_folder(non_street_prev_fldr)
    non_street_images_data = []
    good_image_count = 0

    for filename in os.listdir(input_fldr):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(input_fldr, filename)
        img = io.imread(path)
        if img is None or img.ndim != 3 or img.shape[2] < 3:
            continue
        crop = get_center_crop(img, c_size)
        if crop is None:
            st.warning(f"Skipping {filename}: Crop dimensions are invalid (0 size). Consider larger images or smaller crop_size.")
            continue
        mean_saturation, mean_brightness, max_brightness = analyze_crop(crop)
        is_non_street, reason = is_non_street_image(
            mean_saturation, mean_brightness, max_brightness,
            sat_thresh, v_min, v_max, max_bright_px_thresh
        )
        if is_non_street:
            dest_path = os.path.join(non_street_prev_fldr, filename)
            io.imsave(dest_path, img_as_ubyte(img))
            non_street_images_data.append({"filename": filename, "reason": reason})
        else:
            dest_path = os.path.join(good_imgs_fldr, filename)
            io.imsave(dest_path, img_as_ubyte(img))
            good_image_count += 1
    return non_street_images_data, good_image_count

def main() -> None:
    """
    Main function for the Streamlit UI and image filtering logic.

    :return:
    """
    input_folder = "images/all_images"
    good_images_folder = "images/good_images"
    non_street_preview_folder = "images/temp_non_street_images"
    os.makedirs(good_images_folder, exist_ok=True)
    os.makedirs(non_street_preview_folder, exist_ok=True)

    st.set_page_config(layout="wide", page_title="Image Filter Tuner")
    st.title("Image Filter Parameter Tuner")
    st.write("Adjust the sliders to fine-tune the criteria for identifying 'non-street' images.")

    st.sidebar.header("Filter Parameters")
    sat_threshold = st.sidebar.slider(
        "Saturation Threshold", min_value=0, max_value=255, value=90, step=1,
        help="Images with mean saturation >= this value are considered 'non-street'."
    )
    val_min = st.sidebar.slider(
        "Min Brightness (VAL_MIN)", min_value=0, max_value=255, value=60, step=1,
        help="Images with mean brightness <= this value are considered 'non-street'."
    )
    val_max = st.sidebar.slider(
        "Max Brightness (VAL_MAX)", min_value=0, max_value=255, value=200, step=1,
        help="Images with mean brightness >= this value are considered 'non-street'."
    )
    max_bright_pixel_threshold = st.sidebar.slider(
        "Max Brightness Pixel Threshold", min_value=0, max_value=255, value=25, step=1,
        help="Images where the brightest pixel in the crop is < this value are considered 'non-street'."
    )
    crop_size = st.sidebar.slider(
        "Crop Size (pixels)", min_value=0, max_value=300, value=50, step=1,
        help="Size of the central square crop (e.g., 100 means 100x100 pixels)."
    )
    st.sidebar.info("**Note:** Adjusting these sliders will re-run the analysis and update the results.")

    with st.spinner("Analyzing images..."):
        filtered_images_data, good_img_count = process_images(
            input_folder, good_images_folder, non_street_preview_folder,
            sat_threshold, val_min, val_max, max_bright_pixel_threshold, crop_size
        )

    st.header("Filtering Results")
    st.success(f"**{good_img_count}** images have been classified as 'good' and saved to `{good_images_folder}`.")
    st.write("---")

    st.subheader("Filtered 'Non-Street' Images Preview")
    if filtered_images_data:
        st.info(f"The following {len(filtered_images_data)} images were identified as 'non-street' based on the current parameters. They are temporarily saved in the `{non_street_preview_folder}` folder for your review.")
        cols = st.columns(5)
        for i, img_data in enumerate(filtered_images_data):
            col = cols[i % 5]
            with col:
                st.image(os.path.join(non_street_preview_folder, img_data["filename"]), caption=img_data["filename"], use_container_width=True)
                st.caption(f"Reason: {img_data['reason']}")
    else:
        st.info("No images found matching the current 'non-street' criteria. All images are considered 'good' with the current parameters.")

    st.markdown("---")
    st.write("Once you're satisfied with the parameters, the 'good' images will be in the designated folder.")
    st.write(f"**Current Parameters:** `SAT_THRESHOLD={sat_threshold}`, `VAL_MIN={val_min}`, `VAL_MAX={val_max}`, `MAX_BRIGHT_PIXEL_THRESHOLD={max_bright_pixel_threshold}`, `CROP_SIZE={crop_size}`")

if __name__ == "__main__":
    main()