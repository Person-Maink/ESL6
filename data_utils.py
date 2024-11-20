import numpy as np
from skimage import exposure
from skimage.filters import gaussian
from skimage.transform import resize, rotate
from skimage.util import random_noise
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def build_image_preprocessor(
    flatten: bool = True,
) -> Pipeline:
    """
    Builds an image preprocessing pipeline that includes:
    - Normalizing pixel values to the range [0, 1].
    - Optionally flattening the images for use with dense models.

    Args:
        flatten (bool): Whether to include a step for flattening images into a 1D array (default is True).

    Returns:
        Pipeline: A scikit-learn Pipeline object for preprocessing the image data.
    """

    def normalize_images(images):
        """
        Normalize image pixel values to the range [0, 1].

        Args:
            images (np.ndarray): Array of images to normalize.

        Returns:
            np.ndarray: Array of normalized images.
        """
        return images / 255.0

    def flatten_images(images):
        """
        Flatten images into 1D arrays.

        Args:
            images (np.ndarray): Array of images to flatten.

        Returns:
            np.ndarray: Array of flattened images.
        """
        return images.reshape(images.shape[0], -1)

    # Create transformers for resizing, normalization, and flattening
    normalize_transformer = FunctionTransformer(normalize_images, validate=False)
    flatten_transformer = FunctionTransformer(flatten_images, validate=False)

    # Build the pipeline steps conditionally
    steps = [
        ("normalize", normalize_transformer),  # Normalize pixel values
    ]
    if flatten:
        steps.append(("flatten", flatten_transformer))  # Flatten if specified

    # Construct and return the pipeline
    pipeline = Pipeline(steps=steps)

    return pipeline


def get_augmented_data(
    dataset: tuple[np.ndarray, np.ndarray],
    gen_num: int = 5,
    randome_noise: bool = True,
    rotation: float = 10,
    contrast: tuple[float, float] = (0.5, 99.5),
    gaussian_sigma: tuple[float, float] = (0.75, 0.75),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augments the image dataset by applying various transformations such as:
    - Random contrast adjustment.
    - Adding random noise.
    - Random rotation.
    - Blurring with Gaussian filter.

    Args:
        dataset (tuple): A tuple containing image data (np.ndarray) and their labels (np.ndarray).
        gen_num (int): The number of augmented images to generate for each image (default is 5).
        randome_noise (bool): Whether to apply random noise (default is True).
        rotation (float): The standard deviation for random rotation angles (default is 10 degrees).
        contrast (tuple): Percentile range for contrast adjustment (default is (0.2, 99.8)).
        gaussian_sigma (tuple): Standard deviation for Gaussian blur (default is (1.0, 1.0)).

    Returns:
        tuple: A tuple containing augmented image data and the corresponding labels.
    """

    def augment_image(image):
        """
        Apply augmentations to a single image: contrast adjustment, noise, rotation, and blur.

        Args:
            image (np.ndarray): The input image to augment.

        Returns:
            np.ndarray: The augmented image.
        """
        # Adjust contrast
        v_min, v_max = np.percentile(image, contrast)
        image = exposure.rescale_intensity(image, in_range=(v_min, v_max))

        # Add random noise
        if randome_noise:
            image = random_noise(image)

        # Apply random rotation with bounds
        angle = np.clip(np.random.normal() * rotation, -30, 30)
        image = rotate(image, angle=angle, mode="wrap")

        # Apply Gaussian blurring with bounds
        sigma = np.clip(np.abs(np.random.normal(scale=gaussian_sigma)), 0, 2.0)
        image = gaussian(image, sigma=sigma)

        return image

    # Augment images and labels
    augmented_images = []
    augmented_labels = []

    images, labels = dataset

    for image, label in zip(images, labels):
        augmented_images.extend([augment_image(image) for _ in range(gen_num)])
        augmented_labels.extend([label for _ in range(gen_num)])

    # Combine original and augmented data
    images = np.array(images.tolist() + augmented_images)
    labels = np.array(labels.tolist() + augmented_labels)

    return images, labels


def plot_images_with_labels(dataset, num_images=16, shuffle=True):
    """
    Plots a grid of images along with their corresponding labels.

    Args:
        dataset (tuple): A tuple containing:
            - images (np.ndarray): The input images, expected to be a numpy array of shape
              (n_samples, height, width) or (n_samples, n_features).
            - labels (np.ndarray): The corresponding labels for the images, expected to be
              a 1D numpy array of shape (n_samples,).
        num_images (int, optional): Number of images to plot. Default is 16.
        shuffle (bool, optional): Whether to shuffle the dataset before plotting. Default is True.

    Returns:
        None: Displays the images and their labels in a matplotlib grid plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract images and labels
    images, labels = dataset

    # Ensure num_images doesn't exceed the dataset size
    num_images = min(num_images, len(images))

    # Shuffle the dataset if required
    if shuffle:
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]

    # Determine the grid size
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    # Create the plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i], cmap="gray")  # Display as grayscale
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")  # Remove axis for cleaner look

    # Turn off unused axes
    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
