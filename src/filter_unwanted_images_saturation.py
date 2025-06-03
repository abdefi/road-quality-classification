import streamlit as st
import os
import numpy as np
from skimage import io, color
from skimage.util import img_as_float, img_as_ubyte
import shutil

input_folder = "./images/all_images"
good_images_folder = "./images/good_images"
non_street_preview_folder = "./images/temp_non_street_images"

# Create necessary folders
os.makedirs(good_images_folder, exist_ok=True)
os.makedirs(non_street_preview_folder, exist_ok=True)

st.set_page_config(layout="wide", page_title="Image Filter Tuner")

st.title("📸 Image Filter Parameter Tuner")
st.write("Adjust the sliders to fine-tune the criteria for identifying 'non-street' images.")

st.sidebar.header("Filter Parameters")
sat_threshold = st.sidebar.slider(
    "Saturation Threshold",
    min_value=0, max_value=255, value=90, step=1,
    help="Images with mean saturation >= this value are considered 'non-street'."
)
val_min = st.sidebar.slider(
    "Min Brightness (VAL_MIN)",
    min_value=0, max_value=255, value=60, step=1,
    help="Images with mean brightness <= this value are considered 'non-street'."
)
val_max = st.sidebar.slider(
    "Max Brightness (VAL_MAX)",
    min_value=0, max_value=255, value=200, step=1,
    help="Images with mean brightness >= this value are considered 'non-street'."
)
max_bright_pixel_threshold = st.sidebar.slider(
    "Max Brightness Pixel Threshold",
    min_value=0, max_value=255, value=25, step=1,
    help="Images where the brightest pixel in the crop is < this value are considered 'non-street'."
)
crop_size = st.sidebar.slider(
    "Crop Size (pixels)",
    min_value=50, max_value=300, value=50, step=1,
    help="Size of the central square crop (e.g., 100 means 100x100 pixels)."
)

st.sidebar.info(
    "**Note:** Adjusting these sliders will re-run the analysis and update the results below."
)

@st.cache_data # Cache results to avoid re-reading images if inputs don't change
def process_images(
        input_fldr, good_imgs_fldr, non_street_prev_fldr,
        sat_thresh, v_min, v_max, max_bright_px_thresh, c_size
):
    # Clear previous results in both target folders
    if os.path.exists(good_imgs_fldr):
        shutil.rmtree(good_imgs_fldr)
    os.makedirs(good_imgs_fldr, exist_ok=True)

    if os.path.exists(non_street_prev_fldr):
        shutil.rmtree(non_street_prev_fldr)
    os.makedirs(non_street_prev_fldr, exist_ok=True)

    non_street_images_data = []
    good_image_count = 0

    for filename in os.listdir(input_fldr):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_fldr, filename)
        img = io.imread(path)

        if img is None or img.ndim != 3 or img.shape[2] < 3:
            continue

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        half = c_size // 2
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, cy - half), min(h, cy + half)

        # Ensure crop is not empty
        if x2 <= x1 or y2 <= y1:
            st.warning(f"Skipping {filename}: Crop dimensions are invalid (0 size). Consider larger images or smaller crop_size.")
            continue

        crop = img[y1:y2, x1:x2, :3]

        img_float = img_as_float(crop)
        hsv = color.rgb2hsv(img_float)

        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]

        mean_saturation = np.mean(saturation) * 255
        mean_brightness = np.mean(brightness) * 255
        max_brightness = np.max(brightness) * 255

        # Determine if it's a "non-street" image based on current params
        is_non_street = (
                mean_saturation >= sat_thresh or
                mean_brightness <= v_min or
                mean_brightness >= v_max or
                max_brightness < max_bright_px_thresh
        )

        if is_non_street:
            # Save "non-street" images to the preview folder
            dest_path = os.path.join(non_street_prev_fldr, filename)
            io.imsave(dest_path, img_as_ubyte(img))
            non_street_images_data.append({
                "filename": filename,
                "reason": (
                        (f"Mean Saturation ({mean_saturation:.1f}) >= {sat_thresh}" if mean_saturation >= sat_thresh else "") +
                        (f"Mean Brightness ({mean_brightness:.1f}) <= {v_min}" if mean_brightness <= v_min else "") +
                        (f"Mean Brightness ({mean_brightness:.1f}) >= {v_max}" if mean_brightness >= v_max else "") +
                        (f"Max Brightness ({max_brightness:.1f}) < {max_bright_px_thresh}" if max_brightness < max_bright_px_thresh else "")
                ).strip() # Combine reasons
            })
        else:
            # Save "good" images to the persistent folder
            dest_path = os.path.join(good_imgs_fldr, filename)
            io.imsave(dest_path, img_as_ubyte(img))
            good_image_count += 1

    return non_street_images_data, good_image_count

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