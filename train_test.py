import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import build_image_preprocessor, get_augmented_data
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


def train_test(
    dataset: tuple[np.ndarray, np.ndarray],
    model,
    param_grid: dict,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
    gen_num: int = 5,
    randome_noise: bool = True,
    rotation: float = 10,
    contrast: tuple[float, float] = (0.5, 99.5),
    gaussian_sigma: tuple[float, float] = (1.0, 1.0),
):
    """
    Performs train-validation-test splitting, grid search cross-validation for hyperparameter tuning,
    and final model evaluation on the test set.

    Args:
        dataset (tuple): A tuple containing:
            - X (np.ndarray): Input data (e.g., images), expected to be a numpy array of shape
              (n_samples, height, width) or (n_samples, n_features).
            - y (np.ndarray): Labels corresponding to the input data, expected to be a numpy array
              of shape (n_samples,).
        model (object): A scikit-learn compatible machine learning model (e.g., LogisticRegression, SGDClassifier).
        param_grid (dict): Dictionary specifying the hyperparameters to tune during grid search cross-validation.
        test_size (float, optional): Proportion of the dataset reserved for the test set. Default is 0.2.
        n_splits (int, optional): Number of folds for Stratified K-Fold cross-validation. Default is 5.
        random_state (int, optional): Random state for reproducibility in data splitting and cross-validation.
            Default is 42.
        gen_num (int, optional): Number of augmented samples to generate per original sample. Default is 5.
        randome_noise (bool, optional): Whether to add random noise to the augmented images. Default is True.
        rotation (float, optional): Maximum rotation angle for image augmentation. Default is 10 degrees.
        contrast (tuple[float, float], optional): Percentile range for contrast adjustment. Default is (0.5, 99.5).
        gaussian_sigma (tuple[float, float], optional): Range of Gaussian blur sigma values. Default is (1.0, 1.0).

    Returns:
        tuple: A tuple containing:
            - best_pipeline (Pipeline): The best pipeline found during grid search.
            - best_params (dict): The best hyperparameters from grid search.
            - test_accuracy (float): Accuracy score of the best pipeline on the test set.
    """
    # Step 1: Build the preprocessor
    image_preprocessor = build_image_preprocessor()

    # Step 2: Split dataset into train-validation and test sets
    (X_train_val, y_train_val), (X_test, y_test) = train_validate_test_split(
        dataset, test_size=test_size
    )

    # Step 3: Augment the train_val set
    X_train_val, y_train_val = get_augmented_data(
        (X_train_val, y_train_val),
        gen_num,
        randome_noise,
        rotation,
        contrast,
        gaussian_sigma,
    )

    # Step 4: Perform grid search cross-validation
    grid_search = perform_grid_search_cross_validation(
        (X_train_val, y_train_val),
        image_preprocessor,
        model,
        param_grid,
        n_splits,
        random_state,
    )

    # Step 5: Evaluate the best pipeline on the test set
    best_pipeline = grid_search.best_estimator_
    test_accuracy = evaluate_on_test_set((X_test, y_test), best_pipeline)

    # Output best parameters and cross-validation score
    print("\nBest parameters found:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    return best_pipeline, grid_search.best_params_, test_accuracy


if __name__ == "__main__":
    # Example usage
    model = LogisticRegression(max_iter=1000)  # Logistic Regression baseline model

    labeled_images = np.load("labeled_images.npy")
    labeled_digits = np.load("labeled_digits.npy")

    dataset = (labeled_images, labeled_digits)

    assert len(labeled_images) == len(
        labeled_digits
    ), "Mismatch between images and labels."

    param_grid = {
        "model__C": [0.1, 1, 10],  # Logistic Regression regularization strength
        "model__solver": ["liblinear", "lbfgs"],  # Solvers for Logistic Regression
    }

    best_pipeline, best_params, test_accuracy = train_test(dataset, model, param_grid)

    # Output test set accuracy
    print(f"Test set accuracy: {test_accuracy:.4f}")
