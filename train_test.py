import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import build_image_preprocessor
from model_pipeline import build_image_model_pipeline


def train_validate_test_split(
    dataset: tuple[np.ndarray, np.ndarray],
    test_size: float = 0.2,
    random_state: float = 42,
):
    """
    Splits the dataset into train-validation and test sets.

    Args:
        dataset (tuple[np.ndarray, np.ndarray]): A tuple containing images (X) and labels (y).
        test_size (float, optional): Proportion of the dataset to include in the test set. Default is 0.2.
        random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
        tuple: Train-validation set (X_train_val, y_train_val) and test set (X_test, y_test).
    """
    X, y = dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return (X_train_val, y_train_val), (X_test, y_test)


def perform_grid_search_cross_validation(
    dataset: tuple[np.ndarray, np.ndarray],
    image_preprocessor: Pipeline,
    model,
    param_grid: dict,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Performs grid search cross-validation to find the best hyperparameters.

    Args:
        dataset (tuple[np.ndarray, np.ndarray]): A tuple containing train-validation images (X) and labels (y).
        image_preprocessor (Pipeline): A scikit-learn pipeline for preprocessing image data.
        model (object): A scikit-learn compatible model (e.g., LogisticRegression).
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        n_splits (int, optional): Number of folds for cross-validation. Default is 5.
        random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
        GridSearchCV: A trained GridSearchCV object.
    """
    from sklearn.model_selection import StratifiedKFold

    X, y = dataset

    # Combine preprocessor and model into a single pipeline
    pipeline = build_image_model_pipeline(image_preprocessor, model)

    # Define stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=skf,
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X, y)

    return grid_search


def evaluate_on_test_set(
    test_set: tuple[np.ndarray, np.ndarray], pipeline: Pipeline
) -> float:
    """
    Evaluates the final pipeline on the test set.

    Args:
        test_set (tuple[np.ndarray, np.ndarray]): A tuple containing test images (X) and labels (y).
        pipeline (Pipeline): A trained scikit-learn pipeline.

    Returns:
        float: Accuracy score on the test set.
    """
    X_test, y_test = test_set
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy


if __name__ == "__main__":
    # Step 1: Build the preprocessor and model
    image_preprocessor = build_image_preprocessor()
    model = LogisticRegression(max_iter=1000)  # Logistic Regression baseline model

    # Step 2: Load labeled images and labels
    labeled_images = np.load("labeled_images.npy")
    labeled_digits = np.load("labeled_digits.npy")

    # Validate dataset dimensions
    assert len(labeled_images) == len(
        labeled_digits
    ), "Mismatch between images and labels."

    # Step 3: Split dataset into train-validation and test sets
    (X_train_val, y_train_val), (X_test, y_test) = train_validate_test_split(
        (labeled_images, labeled_digits), test_size=0.2
    )

    # Step 4: Define parameter grid for GridSearchCV
    param_grid = {
        "model__C": [0.1, 1, 10],  # Logistic Regression regularization strength
        "model__solver": ["liblinear", "lbfgs"],  # Solvers for Logistic Regression
    }

    # Step 5: Perform grid search cross-validation
    grid_search = perform_grid_search_cross_validation(
        (X_train_val, y_train_val),
        image_preprocessor,
        model,
        param_grid,
    )

    # Output best parameters and cross-validation score
    print("\nBest parameters found:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    # Step 6: Evaluate the best pipeline on the test set
    best_pipeline = grid_search.best_estimator_
    test_accuracy = evaluate_on_test_set((X_test, y_test), best_pipeline)

    # Output test set accuracy
    print(f"Test set accuracy: {test_accuracy:.4f}")
