import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_utils import build_image_preprocessor


def build_image_model_pipeline(preprocessor: Pipeline, model) -> Pipeline:
    """
    Builds a complete pipeline that includes an image preprocessor and a machine learning model.

    Args:
        preprocessor (Pipeline): The preprocessing pipeline for image data. Should transform raw images into a
            format suitable for the model (e.g., flattened and normalized).
        model (object): A scikit-learn compatible machine learning model (e.g., LogisticRegression).

    Returns:
        Pipeline: A scikit-learn Pipeline object with preprocessing and modeling steps.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # Preprocessing step
            ("model", model),  # Machine learning model step
        ]
    )
    return pipeline


if __name__ == "__main__":
    # Step 1: Build the preprocessor and model
    image_preprocessor = build_image_preprocessor()
    model = LogisticRegression(max_iter=1000)  # Ensure model converges

    # Build the full pipeline
    pipeline = build_image_model_pipeline(image_preprocessor, model)

    # Step 2: Load labeled images and labels
    labeled_images = np.load("labeled_images.npy")
    labeled_digits = np.load("labeled_digits.npy")

    # Validate dataset dimensions
    assert len(labeled_images) == len(
        labeled_digits
    ), "Mismatch between images and labels."

    # Step 3: Train the pipeline
    pipeline.fit(labeled_images, labeled_digits)

    # Step 4: Load autograder images and predict
    autograder_images = np.load("autograder_images.npy")
    predictions = pipeline.predict(autograder_images)

    # Step 5: Estimate accuracy and prepare results
    estimate = np.array(
        [0.7]
    )  # Replace this with your estimate of the accuracy on new data
    result = np.append(estimate, predictions)

    # Step 6: Save results to a file
    pd.DataFrame(result).to_csv("autograder.txt", index=False, header=False)

    # Output predictions shape for verification
    print(f"Predictions shape: {predictions.shape}")
