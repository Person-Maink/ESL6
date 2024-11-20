import numpy as np
from sklearn.linear_model import SGDClassifier

from data_utils import get_augmented_data
from grid_search import get_best_pipeline

if __name__ == "__main__":
    model = SGDClassifier(
        loss="log_loss", max_iter=15000, random_state=42, tol=1e-4, early_stopping=True
    )

    labeled_images = np.load("labeled_images.npy")
    labeled_digits = np.load("labeled_digits.npy")

    assert len(labeled_images) == len(
        labeled_digits
    ), "Mismatch between images and labels."

    dataset = (labeled_images, labeled_digits)

    param_grid = {
        "model__penalty": [
            "l2",
            "l1",
            "elasticnet",
            None,
        ],  # Type of regularization term
        "model__alpha": [0.01, 1, 10, 100],  # Constant of regularization term
    }

    best_pipeline, best_params, test_accuracy = get_best_pipeline(
        dataset, model, param_grid
    )

    # Output test set accuracy
    print(f"Test set accuracy: {test_accuracy:.4f}")
