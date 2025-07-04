from fastai.vision.all import *
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from fastai.learner import Metric
import numpy as np
from collections import Counter


data_path = Path("../../images/dataset")


class BalancedLabelDistanceLoss(nn.Module):
    def __init__(self, class_counts, epsilon=1e-6):
        super().__init__()
        self.register_buffer('weights', 1.0 / (torch.tensor(class_counts).float() + epsilon))
        self.weights = self.weights / self.weights.sum() * len(class_counts)

    def forward(self, preds, targets):
        probs = F.softmax(preds, dim=1)
        class_indices = torch.arange(preds.size(1), device=preds.device).float()
        expected = (probs * class_indices).sum(dim=1)

        error = (expected - targets.float()).pow(2)

        sample_weights = self.weights[targets]
        return error * sample_weights

class BalancedHybridLoss(nn.Module):
    def __init__(self, class_counts, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.label_distance = BalancedLabelDistanceLoss(class_counts)

    def forward(self, preds, targets, **kwargs):
        loss_ce = self.ce(preds, targets)
        loss_ld = self.label_distance(preds, targets)
        combined = self.alpha * loss_ce + (1 - self.alpha) * loss_ld

        if kwargs.get('reduction', None) == 'none':
            return combined
        else:
            return combined.mean()


class BalancedLabelDistance(Metric):
    def __init__(self): self._name = 'balanced_label_distance'

    def reset(self):
        self.class_totals = defaultdict(float)
        self.class_counts = defaultdict(int)

    def accumulate(self, learn):
        preds = learn.pred.argmax(dim=1)
        targets = learn.y

        for pred, target in zip(preds, targets):
            error = abs(pred.item() - target.item())
            self.class_totals[target.item()] += error
            self.class_counts[target.item()] += 1

    @property
    def value(self):
        class_avg_distances = []
        for cls in sorted(self.class_counts.keys()):
            count = self.class_counts[cls]
            if count > 0:
                avg = self.class_totals[cls] / count
                class_avg_distances.append(avg)
        return sum(class_avg_distances) / len(class_avg_distances) if class_avg_distances else None



def main():
    dls = ImageDataLoaders.from_folder(
        data_path,
        train='train',
        valid='val',
        valid_pct=None,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(mult=1.5) + [Normalize.from_stats(*imagenet_stats)],
        bs=64,
        num_workers=0,
        shuffle=True
    )
    print(f"Number of training batches: {len(dls.train)}")

    train_folder = data_path / 'train'
    train_labels = [int(path.parent.name) for path in train_folder.rglob('*') if path.is_file()]
    num_classes = len(set(train_labels))
    counts = Counter(train_labels)
    class_counts = [counts.get(i, 1) for i in range(num_classes)]
    print("Class counts:", class_counts)
    learn = vision_learner(dls, resnet50,
        loss_func=BalancedHybridLoss(class_counts, alpha=0.35),
        metrics=[
            accuracy,
            BalancedLabelDistance()
        ],
        cbs=[
            SaveModelCallback(monitor='balanced_label_distance', comp=np.less, fname='best_model'),
            EarlyStoppingCallback(monitor='valid_loss', comp=np.less, patience=5),
            ReduceLROnPlateau(monitor='valid_loss', patience=2),
            GradientClip(max_norm=1.0),
            CSVLogger(fname='training_log.csv')
        ])
    print("Created learner")
    learn.fine_tune(40)
    print("Fine-tuned model")

    val_predictions, val_targets = learn.get_preds()
    val_accuracy = accuracy_score(val_targets, val_predictions.argmax(dim=1))
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")


    class_totals = defaultdict(float)
    class_counts = defaultdict(int)
    for pred, target in zip(val_predictions.argmax(dim=1), val_targets):
        dist = abs(pred.item() - target.item())
        class_totals[target.item()] += dist
        class_counts[target.item()] += 1
    class_avg_dists = []
    for cls in sorted(class_counts.keys()):
        avg_dist = class_totals[cls] / class_counts[cls]
        class_avg_dists.append(avg_dist)
        print(f"Class {cls}: Avg label distance = {avg_dist:.4f}")
    balanced_distance = sum(class_avg_dists) / len(class_avg_dists)
    print(f"\nBalanced Label Distance (validation): {balanced_distance:.4f}")


    preds, targs = learn.get_preds(dl=learn.dls.valid)
    decoded = preds.argmax(dim=1)
    assert len(decoded) == len(targs), f"Shape mismatch: {len(decoded)} vs {len(targs)}"
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(targs.cpu(), decoded.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=learn.dls.vocab)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.tight_layout()
    plt.show()

    print("\nPer-class accuracy:")
    cm = confusion_matrix(val_targets, val_predictions.argmax(dim=1))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for cls, acc in zip(learn.dls.vocab, per_class_acc):
        print(f"{cls}: {acc:.4f}")

    print("Evaluation completed")

    learn.remove_cb(CSVLogger)
    learn.export("road_quality_classifier.pkl")
    print("Model saved successfully")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()