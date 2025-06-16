# road-quality-classification

This repository offers a complete pipeline to **automatically assess road quality using Google Street View images**. From fetching the images to advanced deep learning classification, we've got it covered.

## Key Features

* **Image Extraction:** Programmatically pulls Google Street View images for any specified geographic area.
* **Smart Filtering:** Filters out non-street images (e.g., sky, ground) with an interactive UI for precise control.
* **Deep Learning Classification:** Trains a **ResNet50 CNN** to classify road quality, using custom loss and metrics (`BalancedHybridLoss`, `BalancedLabelDistance`) to handle ordinal data and class imbalance effectively.
* **Map Visualization:** Generates an **interactive HTML map** with colored markers to visually represent classified road quality data, making spatial analysis intuitive.
* **Comprehensive Evaluation:** Provides detailed accuracy and label distance metrics, including per-class performance.

---

## How It Works

The process is divided into three main stages, followed by visualization:

1.  **Image Extraction:** Downloads unique Street View panoramas, creating a grid of coordinates for a defined area.
2.  **Image Categorization (Filtering):** Uses a Streamlit app to interactively filter out irrelevant images based on visual properties like saturation and brightness.
3.  **Road Quality Classification:** Trains a deep learning model on the filtered images to classify road quality. Custom functions ensure robust training even with imbalanced datasets.
4.  **Map Visualization:** After classification, a Python script generates an HTML map, overlaying colored circles at image locations to show predicted road quality, offering a clear spatial overview.

## Getting Started

### Configuration

Create a `.env` file in the root of your project directory and add the following:

```bash
GCP_API_KEY="YOUR_GOOGLE_CLOUD_API_KEY"
GCP_SIGNING_SECRET="YOUR_GOOGLE_CLOUD_SIGNING_SECRET"

NORTH_WEST="LATITUDE_NW,LONGITUDE_NW"
SOUTH_EAST="LATITUDE_SE,LONGITUDE_SE"
```

Replace placeholders with your actual Google Cloud credentials and desired geographical bounding box coordinates.
The `NORTH_WEST` and `SOUTH_EAST` variables define the bounding box for the area from which you want to extract street view images.

### [Optional] Download Dataset

If you want to use our pre-existing dataset for training and evaluation, you can download it from [here.](https://drive.google.com/file/d/1x3EScOoQ9fMtsAY1nGyI0p9BcaqriR1J/view?usp=sharing)

This dataset is based on the work of [lenoch0d.](https://github.com/lenoch0d/road-quality-classification)


---

## Usage Guide

This guide outlines the steps to extract, categorize, classify, and visualize road quality data using the provided scripts.

### 1. Run Image Extraction

This step downloads street view images to a specified output directory.

```bash
python ./src/fetch/extract_street_view_images.py
```

* **Output:** Images will be saved in the `out/images/` directory.

### 2. Run Image Categorization (Streamlit App)

```bash

streamlit run ./src/filter/filter_unwanted_images_saturation.py
```

* Open your browser to the URL displayed in your terminal.
* **Action:** Adjust parameters within the Streamlit app to review and filter images.
* **Output:** The "good" (filtered) images will be saved in `images/good_images/`.

### 3. Train Road Quality Classifier

This step trains a ResNet50 model using a pre-existing dataset.

* **Dataset:** This project utilizes the dataset from the [road-quality-classification](https://github.com/lenoch0d/road-quality-classification) repository.
* **Setup:** Add the dataset to your project and ensure the path is correctly configured in the `src/model/train_model.py` script.

```bash
python train_classifier.py
```

* **Output:**
    * The best-performing model will be saved as `road_quality_classifier.pkl`.
    * Training progress will be logged in `training_log.csv`.

### 4. Run Road Quality Classification

Before running the classification, verify that the image folder paths in the `src/model/rate_pictures.py` script are set to match your local setup.

```bash
python src/model/rate_pictures.py
```

* **Output:** Images will be categorized and saved into subfolders (0 to 5) within the output directory, corresponding to their predicted road quality class.

### 5. Generate Map Visualization

After classifying the images, run this script to create an interactive HTML map.

```bash
python src/map/generate_map_from_images.py
```

* **Action:** Open the generated `map.html` file in your web browser.
* **Output:** An interactive HTML map visualizing the classified road quality data.
