import numpy as np
from sklearn.linear_model import SGDClassifier

from data_preprocessing import build_image_preprocessor
from train_test import (
    evaluate_on_test_set,
    perform_grid_search_cross_validation,
    train_validate_test_split,
)

#     # Step 4: Define parameter grid for GridSearchCV
#     param_grid = {
#         "model__C": [0.1, 1, 10],  # Logistic Regression regularization strength
#         "model__solver": ["liblinear", "lbfgs"],  # Solvers for Logistic Regression
#     }

#     # Step 5: Perform grid search cross-validation
#     grid_search = perform_grid_search_cross_validation(
#         (X_train_val, y_train_val),
#         image_preprocessor,
#         model,
#         param_grid,
#     )

#     # Output best parameters and cross-validation score
#     print("\nBest parameters found:", grid_search.best_params_)
#     print("Best cross-validation accuracy:", grid_search.best_score_)

#     # Step 6: Evaluate the best pipeline on the test set
#     best_pipeline = grid_search.best_estimator_
#     test_accuracy = evaluate_on_test_set((X_test, y_test), best_pipeline)

#     # Output test set accuracy
#     print(f"Test set accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    # Step 1: Build the preprocessor and model
    image_preprocessor = build_image_preprocessor()
    model = SGDClassifier(loss="log_loss", max_iter=15000, random_state=42, tol=1e-4)

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
        # "model__penalty": [
        #     "l2",
        #     "l1",
        #     "elasticnet",
        #     None,
        # ],  # Type of regularization term
        "model__alpha": [0.01, 1, 10, 100],  # Constant of regularization term
        "model__learning_rate": [
            "constant",
            "optimal",
            "invscaling",
            "adaptive",
        ],  # Learning rate strategies
        "model__eta0": [
            0.0001,
            0.001,
            0.01,
        ],  # Initial learning rate for 'constant' or 'invscaling'
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
