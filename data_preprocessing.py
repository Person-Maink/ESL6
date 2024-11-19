import numpy as np
from skimage import exposure
from skimage.filters import gaussian
from skimage.transform import resize, rotate
from skimage.util import random_noise
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def build_image_preprocessor(
    rescale: tuple[int, int] = (28, 28),
    flatten: bool = True,
) -> Pipeline:
    """
    Builds an image preprocessing pipeline that includes:
    - Rescaling images to the given size.
    - Normalizing pixel values to the range [0, 1].
    - Optionally flattening the images for use with dense models.

    Args:
        rescale (tuple): A tuple specifying the desired target size for image rescaling (default is (28, 28)).
        flatten (bool): Whether to include a step for flattening images into a 1D array (default is True).

    Returns:
        Pipeline: A scikit-learn Pipeline object for preprocessing the image data.
    """

    def resize_images(images):
        """
        Rescale images to the desired size.

        Args:
            images (np.ndarray): Array of images to resize.

        Returns:
            np.ndarray: Array of resized images.
        """
        return np.array([resize(img, rescale, anti_aliasing=True) for img in images])

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
    resize_transformer = FunctionTransformer(resize_images, validate=False)
    normalize_transformer = FunctionTransformer(normalize_images, validate=False)
    flatten_transformer = FunctionTransformer(flatten_images, validate=False)

    # Build the pipeline steps conditionally
    steps = [
        ("resize", resize_transformer),  # Rescale to the specified size
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
    contrast: tuple[float, float] = (0.2, 99.8),
    gaussian_sigma: tuple[float, float] = (1.0, 1.0),
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

        # Apply random rotation
        angle = np.random.normal() * rotation
        image = rotate(image, angle=angle)

        # Apply Gaussian blurring
        image = gaussian(image, sigma=gaussian_sigma)

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


if __name__ == "__main__":
    # Load image dataset and augment data
    labeled_images = np.load("labeled_images.npy")
    labeled_digits = np.load("labeled_digits.npy")

    print(f"Original dataset shapes: {labeled_images.shape}, {labeled_digits.shape}")

    # Generate augmented dataset
    augmented_images, augmented_labels = get_augmented_data(
        (labeled_images, labeled_digits)
    )

    print(
        f"Augmented dataset shapes: {augmented_images.shape}, {augmented_labels.shape}"
    )
