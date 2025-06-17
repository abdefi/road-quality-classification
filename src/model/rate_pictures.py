from fastai.vision.all import *
import torch
from torch.nn.functional import softmax
from PIL import Image
from train_model import BalancedLabelDistanceLoss, BalancedHybridLoss

def main():
    # === Load the model ===
    learn_inf = load_learner('road_quality_classifier_727_3616_custom.pkl')
    print("Vocab:", learn_inf.dls.vocab)

    # === Set paths ===
    source_folder = Path("../../images/good_images")
    destination_base = Path("../../images/sorted_by_class")
    destination_base.mkdir(exist_ok=True, parents=True)

    # === Allowed image extensions ===
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # === Process each image ===
    for img_path in source_folder.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue
        try:
            # Skip unreadable or corrupt images
            Image.open(img_path).verify()

            # Predict
            dl = learn_inf.dls.test_dl([img_path])
            xb = dl.one_batch()[0]
            preds = learn_inf.model(xb)
            probs = softmax(preds[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_class = learn_inf.dls.vocab[pred_idx] if pred_idx < len(learn_inf.dls.vocab) else f"Invalid({pred_idx})"
            confidence = probs[pred_idx].item()

            print(f"{img_path.name}: Predicted class {pred_class} (prob: {confidence:.4f})")

            # Copy to class subfolder
            target_folder = destination_base / str(pred_class)
            target_folder.mkdir(exist_ok=True, parents=True)
            shutil.copy(img_path, target_folder / img_path.name)

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")


if __name__ == "__main__":
    main()