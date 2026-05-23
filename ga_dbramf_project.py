"""
Impulse Noise Removal with Edge Preservation
Author: Uzeyir Yaman
Student ID: 2111011074

This script contains the complete Google Colab / Python implementation used for:
- benchmark image loading
- impulse noise generation
- standard Median Filter
- standard DBRAMF
- proposed GA-DBRAMF
- quantitative evaluation with MSE, PSNR, SSIM, and Pratt's FOM
- full experimental matrix
- metric plots
- ablation study

The code comments and variable names are intentionally written in English for GitHub submission.
"""

# ============================================================
# 1. Imports and experiment configuration
# ============================================================

from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data
from skimage.metrics import structural_similarity as ssim


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path("/content/ga_dbramf_project") if Path("/content").exists() else Path("ga_dbramf_project")
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
DATA_DIR = BASE_DIR / "data"

for directory in [FIGURES_DIR, RESULTS_DIR, TABLES_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project folders are ready:")
print("Base directory:", BASE_DIR)
print("Figures directory:", FIGURES_DIR)
print("Results directory:", RESULTS_DIR)
print("Tables directory:", TABLES_DIR)


# ============================================================
# 2. General helper functions
# ============================================================

def ensure_uint8_grayscale(image, target_size=(512, 512)):
    """Convert an image to grayscale uint8 format and resize it."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image.astype(np.uint8)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image.astype(np.uint8)


def load_lena_image(target_size=(512, 512)):
    """
    Load the Lena benchmark image.

    The function first searches for a local Lena file. If it cannot find one,
    it downloads a commonly used OpenCV sample image.
    """
    possible_paths = [
        Path("lena.png"), Path("lena.jpg"), Path("lenna.png"), Path("lenna.jpg"),
        DATA_DIR / "lena.jpg", DATA_DIR / "lena.png",
        Path("/content/lena.png"), Path("/content/lena.jpg"),
        Path("/content/lenna.png"), Path("/content/lenna.jpg"),
    ]

    for image_path in possible_paths:
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                print("Lena image loaded from:", image_path)
                return ensure_uint8_grayscale(image, target_size=target_size)

    download_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    downloaded_path = DATA_DIR / "lena.jpg"

    try:
        print("Local Lena image was not found. Downloading Lena sample image...")
        urllib.request.urlretrieve(download_url, downloaded_path)
        image = cv2.imread(str(downloaded_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError("Downloaded Lena image could not be read.")
        return ensure_uint8_grayscale(image, target_size=target_size)
    except Exception as error:
        raise RuntimeError(
            "Lena image could not be loaded. Please place lena.jpg or lena.png "
            "in the project folder and run the script again."
        ) from error


def load_cameraman_image(target_size=(512, 512)):
    """Load the Cameraman benchmark image from scikit-image."""
    image = data.camera()
    print("Cameraman image loaded from scikit-image.")
    return ensure_uint8_grayscale(image, target_size=target_size)


def describe_image(image_name, image):
    """Return basic image statistics as a dictionary."""
    return {
        "Image": image_name,
        "Height": image.shape[0],
        "Width": image.shape[1],
        "Data Type": str(image.dtype),
        "Minimum Intensity": int(np.min(image)),
        "Maximum Intensity": int(np.max(image)),
        "Mean Intensity": float(np.mean(image)),
        "Standard Deviation": float(np.std(image)),
    }


def show_image_grid(images, titles, cols=4, figsize=(16, 8), save_path=None):
    """Display and optionally save a grid of grayscale images."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
            axis.set_title(titles[index], fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def calculate_changed_pixel_ratio(original_image, noisy_image):
    """Calculate the ratio of changed pixels between original and noisy images."""
    changed_pixels = np.sum(original_image != noisy_image)
    total_pixels = original_image.size
    return float(changed_pixels / total_pixels)


# ============================================================
# 3. Impulse noise generation functions
# ============================================================

def add_salt_and_pepper_noise(image, density=0.3, random_seed=42):
    """Add Salt-and-Pepper impulse noise to a grayscale image."""
    rng = np.random.default_rng(random_seed)
    noisy_image = image.copy().astype(np.uint8)

    total_pixels = image.size
    number_of_corrupted_pixels = int(round(density * total_pixels))

    flat_indices = rng.choice(total_pixels, size=number_of_corrupted_pixels, replace=False)
    number_of_salt_pixels = number_of_corrupted_pixels // 2

    salt_indices = flat_indices[:number_of_salt_pixels]
    pepper_indices = flat_indices[number_of_salt_pixels:]

    noisy_flat = noisy_image.reshape(-1)
    noisy_flat[salt_indices] = 255
    noisy_flat[pepper_indices] = 0

    return noisy_flat.reshape(image.shape).astype(np.uint8)


def add_random_valued_impulse_noise(image, density=0.3, random_seed=42):
    """Add Random-Valued Impulse Noise (RVIN) to a grayscale image."""
    rng = np.random.default_rng(random_seed)
    noisy_image = image.copy().astype(np.uint8)

    total_pixels = image.size
    number_of_corrupted_pixels = int(round(density * total_pixels))

    flat_indices = rng.choice(total_pixels, size=number_of_corrupted_pixels, replace=False)
    random_values = rng.integers(0, 256, size=number_of_corrupted_pixels, dtype=np.uint8)

    noisy_flat = noisy_image.reshape(-1)
    noisy_flat[flat_indices] = random_values

    return noisy_flat.reshape(image.shape).astype(np.uint8)


def add_structured_burst_noise(image, density=0.3, block_size=16, random_seed=42):
    """
    Add Structured Burst Noise by corrupting randomly selected square blocks.

    Inside selected blocks, pixels are assigned random binary impulse values
    0 or 255. The function returns both the noisy image and the corruption mask.
    """
    rng = np.random.default_rng(random_seed)
    noisy_image = image.copy().astype(np.uint8)
    corrupted_mask = np.zeros(image.shape, dtype=bool)

    height, width = image.shape
    target_corrupted_pixels = int(round(density * image.size))

    max_attempts = 100000
    attempts = 0

    while np.sum(corrupted_mask) < target_corrupted_pixels and attempts < max_attempts:
        attempts += 1
        x = rng.integers(0, max(1, height - block_size + 1))
        y = rng.integers(0, max(1, width - block_size + 1))

        x_end = min(x + block_size, height)
        y_end = min(y + block_size, width)

        block_shape = (x_end - x, y_end - y)
        random_binary_values = rng.choice([0, 255], size=block_shape).astype(np.uint8)

        noisy_image[x:x_end, y:y_end] = random_binary_values
        corrupted_mask[x:x_end, y:y_end] = True

    return noisy_image.astype(np.uint8), corrupted_mask


# ============================================================
# 4. Standard Median Filter baseline
# ============================================================

def apply_standard_median_filter(noisy_image, kernel_size=3):
    """Apply the standard median filter using OpenCV."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number.")
    return cv2.medianBlur(noisy_image.astype(np.uint8), kernel_size).astype(np.uint8)


# ============================================================
# 5. Standard DBRAMF baseline
# ============================================================

def extract_window(image, x, y, window_size):
    """Extract a local window around a pixel."""
    half_size = window_size // 2
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, image.shape[0])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, image.shape[1])
    return image[x_min:x_max, y_min:y_max]


def get_healthy_binary_pixels(window):
    """Return pixels that are not 0 or 255 under binary impulse noise assumption."""
    return window[(window > 0) & (window < 255)]


def apply_standard_dbramf(noisy_image, max_window_size=5):
    """
    Apply standard Decision-Based Recursive Adaptive Median Filter (DBRAMF).

    If a corrupted pixel has no healthy neighbor in the maximum window, the
    previous processed pixel is used as the fallback value.
    """
    if max_window_size not in [3, 5]:
        raise ValueError("max_window_size must be either 3 or 5.")

    restored_image = noisy_image.copy().astype(np.uint8)
    height, width = noisy_image.shape

    for x in range(height):
        for y in range(width):
            current_pixel = int(restored_image[x, y])

            if 0 < current_pixel < 255:
                continue

            replacement_value = None

            window_3 = extract_window(restored_image, x, y, window_size=3)
            healthy_pixels_3 = get_healthy_binary_pixels(window_3)

            if healthy_pixels_3.size > 0:
                replacement_value = np.median(healthy_pixels_3)
            elif max_window_size == 5:
                window_5 = extract_window(restored_image, x, y, window_size=5)
                healthy_pixels_5 = get_healthy_binary_pixels(window_5)

                if healthy_pixels_5.size > 0:
                    replacement_value = np.median(healthy_pixels_5)

            if replacement_value is None:
                if y > 0:
                    replacement_value = restored_image[x, y - 1]
                elif x > 0:
                    replacement_value = restored_image[x - 1, width - 1]
                else:
                    replacement_value = 128

            restored_image[x, y] = np.uint8(np.clip(round(replacement_value), 0, 255))

    return restored_image.astype(np.uint8)


# ============================================================
# 6. Proposed GA-DBRAMF
# ============================================================

def compute_binary_impulse_mask(noisy_image):
    """Compute corrupted mask for binary impulse noise."""
    return (noisy_image == 0) | (noisy_image == 255)


def compute_rvin_outlier_mask(noisy_image, threshold=45, median_kernel_size=3):
    """
    Detect RVIN-like outliers using local median deviation.

    A pixel is considered suspicious if its absolute deviation from the local
    median is larger than the selected threshold.
    """
    local_median = cv2.medianBlur(noisy_image.astype(np.uint8), median_kernel_size)
    difference = np.abs(noisy_image.astype(np.int16) - local_median.astype(np.int16))
    corrupted_mask = difference > threshold
    return corrupted_mask


def apply_ga_dbramf_balanced(
    noisy_image,
    max_window_size=5,
    detection_mode="binary",
    rvin_threshold=45,
    return_stats=False,
):
    """
    Apply the proposed Gradient-Aware DBRAMF.

    Main idea:
    - Keep originally healthy pixels unchanged.
    - Use 3x3 and 5x5 median of reliable pixels when available.
    - If no reliable pixel exists in the maximum window, use gradient-aware
      directional interpolation instead of previous-pixel fallback.
    """
    if max_window_size not in [3, 5]:
        raise ValueError("max_window_size must be either 3 or 5.")

    restored_image = noisy_image.copy().astype(np.uint8)

    if detection_mode == "binary":
        original_corrupted_mask = compute_binary_impulse_mask(noisy_image)
    elif detection_mode == "rvin":
        original_corrupted_mask = compute_rvin_outlier_mask(
            noisy_image,
            threshold=rvin_threshold,
            median_kernel_size=3,
        )
    else:
        raise ValueError("detection_mode must be either 'binary' or 'rvin'.")

    reliable_mask = ~original_corrupted_mask.copy()
    height, width = noisy_image.shape

    stats = {
        "total_pixels": int(height * width),
        "detected_corrupted_pixels": int(np.sum(original_corrupted_mask)),
        "kept_healthy_pixels": int(np.sum(~original_corrupted_mask)),
        "replaced_by_3x3_median": 0,
        "replaced_by_5x5_median": 0,
        "replaced_by_gradient_interpolation": 0,
    }

    def get_window_bounds(x, y, window_size):
        half = window_size // 2
        x_min = max(x - half, 0)
        x_max = min(x + half + 1, height)
        y_min = max(y - half, 0)
        y_max = min(y + half + 1, width)
        return x_min, x_max, y_min, y_max

    def find_nearest_reliable(x, y, direction, sign, max_radius=25):
        dx, dy = direction
        for radius in range(1, max_radius + 1):
            nx = x + sign * radius * dx
            ny = y + sign * radius * dy

            if nx < 0 or nx >= height or ny < 0 or ny >= width:
                break

            if reliable_mask[nx, ny]:
                return int(restored_image[nx, ny])

        return None

    def gradient_interpolate(x, y):
        directions = {
            "0_degrees": (0, 1),
            "45_degrees": (-1, 1),
            "90_degrees": (-1, 0),
            "135_degrees": (-1, -1),
        }

        candidates = []

        for direction_vector in directions.values():
            value_negative = find_nearest_reliable(x, y, direction_vector, sign=-1, max_radius=25)
            value_positive = find_nearest_reliable(x, y, direction_vector, sign=1, max_radius=25)

            if value_negative is not None and value_positive is not None:
                gradient_value = abs(value_negative - value_positive)
                interpolated_value = (value_negative + value_positive) / 2.0
                candidates.append((gradient_value, interpolated_value))

        if len(candidates) > 0:
            _, best_value = min(candidates, key=lambda item: item[0])
            return int(np.clip(round(best_value), 0, 255))

        x_min, x_max, y_min, y_max = get_window_bounds(x, y, 7)
        local_values = restored_image[x_min:x_max, y_min:y_max][reliable_mask[x_min:x_max, y_min:y_max]]

        if local_values.size > 0:
            return int(np.clip(round(np.median(local_values)), 0, 255))

        if y > 0:
            return int(restored_image[x, y - 1])
        if x > 0:
            return int(restored_image[x - 1, width - 1])
        return 128

    for x in range(height):
        for y in range(width):
            if not original_corrupted_mask[x, y]:
                continue

            replacement_value = None

            x_min, x_max, y_min, y_max = get_window_bounds(x, y, 3)
            window_values = restored_image[x_min:x_max, y_min:y_max]
            window_mask = original_corrupted_mask[x_min:x_max, y_min:y_max]
            valid_pixels = window_values[~window_mask]

            if valid_pixels.size > 0:
                replacement_value = np.median(valid_pixels)
                stats["replaced_by_3x3_median"] += 1
            elif max_window_size == 5:
                x_min, x_max, y_min, y_max = get_window_bounds(x, y, 5)
                window_values = restored_image[x_min:x_max, y_min:y_max]
                window_mask = original_corrupted_mask[x_min:x_max, y_min:y_max]
                valid_pixels = window_values[~window_mask]

                if valid_pixels.size > 0:
                    replacement_value = np.median(valid_pixels)
                    stats["replaced_by_5x5_median"] += 1

            if replacement_value is None:
                replacement_value = gradient_interpolate(x, y)
                stats["replaced_by_gradient_interpolation"] += 1

            restored_image[x, y] = np.uint8(np.clip(round(replacement_value), 0, 255))
            reliable_mask[x, y] = True

    if return_stats:
        return restored_image.astype(np.uint8), stats

    return restored_image.astype(np.uint8)


# ============================================================
# 7. Evaluation metrics
# ============================================================

def calculate_mse(original_image, restored_image):
    """Calculate Mean Squared Error."""
    original_float = original_image.astype(np.float64)
    restored_float = restored_image.astype(np.float64)
    return float(np.mean((original_float - restored_float) ** 2))


def calculate_psnr(original_image, restored_image, max_pixel_value=255):
    """Calculate Peak Signal-to-Noise Ratio in decibels."""
    mse_value = calculate_mse(original_image, restored_image)
    if mse_value == 0:
        return float("inf")
    return float(10 * np.log10((max_pixel_value ** 2) / mse_value))


def calculate_ssim_metric(original_image, restored_image):
    """Calculate Structural Similarity Index."""
    return float(ssim(original_image.astype(np.uint8), restored_image.astype(np.uint8), data_range=255))


def extract_edge_map(image, low_threshold=50, high_threshold=150):
    """Extract a binary edge map using Canny edge detector."""
    edges = cv2.Canny(image.astype(np.uint8), threshold1=low_threshold, threshold2=high_threshold)
    return edges > 0


def calculate_pratt_fom(original_image, restored_image, alpha=1/9, low_threshold=50, high_threshold=150):
    """Calculate Pratt's Figure of Merit using Canny edge maps."""
    original_edges = extract_edge_map(original_image, low_threshold=low_threshold, high_threshold=high_threshold)
    restored_edges = extract_edge_map(restored_image, low_threshold=low_threshold, high_threshold=high_threshold)

    number_of_ideal_edges = int(np.sum(original_edges))
    number_of_detected_edges = int(np.sum(restored_edges))

    if number_of_ideal_edges == 0 and number_of_detected_edges == 0:
        return 1.0
    if number_of_ideal_edges == 0 or number_of_detected_edges == 0:
        return 0.0

    distance_transform = ndimage.distance_transform_edt(~original_edges)
    detected_distances = distance_transform[restored_edges]
    fom_values = 1.0 / (1.0 + alpha * (detected_distances ** 2))
    fom_value = np.sum(fom_values) / max(number_of_ideal_edges, number_of_detected_edges)

    return float(fom_value)


def calculate_all_metrics(original_image, restored_image):
    """Calculate MSE, PSNR, SSIM, and Pratt's FOM."""
    return {
        "MSE": calculate_mse(original_image, restored_image),
        "PSNR": calculate_psnr(original_image, restored_image),
        "SSIM": calculate_ssim_metric(original_image, restored_image),
        "Pratt_FOM": calculate_pratt_fom(original_image, restored_image),
    }


# ============================================================
# 8. Full experimental matrix helper functions
# ============================================================

def generate_noisy_image(image, noise_type, density, random_seed):
    """Generate a noisy image according to the selected noise type."""
    if noise_type == "salt_and_pepper":
        return add_salt_and_pepper_noise(image, density=density, random_seed=random_seed)

    if noise_type == "rvin":
        return add_random_valued_impulse_noise(image, density=density, random_seed=random_seed)

    if noise_type == "structured_burst":
        noisy_image, _ = add_structured_burst_noise(
            image,
            density=density,
            block_size=16,
            random_seed=random_seed,
        )
        return noisy_image

    raise ValueError("Unknown noise type.")


def apply_restoration_method(noisy_image, method_name, noise_type):
    """Apply a selected restoration method to a noisy image."""
    if method_name == "standard_median_filter":
        return apply_standard_median_filter(noisy_image, kernel_size=3)

    if method_name == "standard_dbramf":
        return apply_standard_dbramf(noisy_image, max_window_size=5)

    if method_name == "ga_dbramf":
        if noise_type == "rvin":
            return apply_ga_dbramf_balanced(
                noisy_image,
                max_window_size=5,
                detection_mode="rvin",
                rvin_threshold=45,
            )

        return apply_ga_dbramf_balanced(
            noisy_image,
            max_window_size=5,
            detection_mode="binary",
        )

    raise ValueError("Unknown restoration method.")


def evaluate_ablation_case(original_image, restored_image, case_name):
    """Calculate all metrics for an ablation experiment case."""
    metrics = calculate_all_metrics(original_image, restored_image)
    return {
        "Case": case_name,
        "MSE": metrics["MSE"],
        "PSNR": metrics["PSNR"],
        "SSIM": metrics["SSIM"],
        "Pratt_FOM": metrics["Pratt_FOM"],
    }


# ============================================================
# 9. Plotting helper functions
# ============================================================

def plot_metric_curves(plot_results_df, metric_column, metric_label, save_filename):
    """Plot metric curves versus noise density for each noise type."""
    method_order = ["Standard Median Filter", "Standard DBRAMF", "GA-DBRAMF"]
    noise_type_order = ["Salt-and-Pepper", "RVIN", "Structured Burst"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for axis_index, noise_type in enumerate(noise_type_order):
        ax = axes[axis_index]
        noise_subset = plot_results_df[plot_results_df["Noise_Type"] == noise_type]

        for method_name in method_order:
            method_subset = noise_subset[noise_subset["Method"] == method_name].sort_values("Density_Percent")

            ax.plot(
                method_subset["Density_Percent"],
                method_subset[metric_column],
                marker="o",
                linewidth=2,
                label=method_name,
            )

        ax.set_title(noise_type)
        ax.set_xlabel("Noise Density (%)")
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks([10, 30, 50, 70, 80])

    axes[0].legend(loc="best")
    plt.tight_layout()

    save_path = FIGURES_DIR / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"{metric_label} curve figure saved to:")
    print(save_path)


def plot_average_metric_comparison(average_by_method):
    """Plot average PSNR, SSIM, and Pratt FOM comparison by method."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics_to_plot = [
        ("PSNR", "Average PSNR (dB)"),
        ("SSIM", "Average SSIM"),
        ("Pratt_FOM", "Average Pratt's FOM"),
    ]

    for idx, (metric_column, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx]
        sorted_df = average_by_method.sort_values(metric_column, ascending=False)
        ax.bar(sorted_df["Method"], sorted_df[metric_column])
        ax.set_title(metric_label)
        ax.set_ylabel(metric_label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = FIGURES_DIR / "42.JPG"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Average metric comparison figure saved to:")
    print(save_path)


def plot_runtime_comparison(final_summary_df):
    """Plot average runtime comparison by method."""
    plt.figure(figsize=(7, 4))
    plt.bar(final_summary_df["Method"], final_summary_df["Average_Runtime_Seconds"])
    plt.ylabel("Average Runtime (seconds)")
    plt.xlabel("Method")
    plt.title("Average Runtime Comparison")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    save_path = FIGURES_DIR / "49.JPG"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Runtime comparison figure saved to:")
    print(save_path)


# ============================================================
# 10. Main experiment pipeline
# ============================================================

def main():
    """Run the complete experimental pipeline."""

    # --------------------------------------------------------
    # Load benchmark images
    # --------------------------------------------------------
    lena_image = load_lena_image()
    cameraman_image = load_cameraman_image()

    image_properties_df = pd.DataFrame([
        describe_image("Lena", lena_image),
        describe_image("Cameraman", cameraman_image),
    ])
    image_properties_df.to_csv(TABLES_DIR / "benchmark_image_properties.csv", index=False)
    print(image_properties_df)

    show_image_grid(
        images=[lena_image, cameraman_image],
        titles=["Original Lena", "Original Cameraman"],
        cols=2,
        figsize=(10, 4),
        save_path=FIGURES_DIR / "6.JPG",
    )

    # --------------------------------------------------------
    # Generate representative noisy images
    # --------------------------------------------------------
    lena_sp_noise = add_salt_and_pepper_noise(lena_image, density=0.30, random_seed=RANDOM_SEED)
    lena_rvin_noise = add_random_valued_impulse_noise(lena_image, density=0.30, random_seed=RANDOM_SEED)
    lena_burst_noise, _ = add_structured_burst_noise(lena_image, density=0.30, block_size=16, random_seed=RANDOM_SEED)

    cameraman_sp_noise = add_salt_and_pepper_noise(cameraman_image, density=0.30, random_seed=RANDOM_SEED)
    cameraman_rvin_noise = add_random_valued_impulse_noise(cameraman_image, density=0.30, random_seed=RANDOM_SEED)
    cameraman_burst_noise, _ = add_structured_burst_noise(cameraman_image, density=0.30, block_size=16, random_seed=RANDOM_SEED)

    lena_sp_noise_80 = add_salt_and_pepper_noise(lena_image, density=0.80, random_seed=RANDOM_SEED)
    lena_rvin_noise_80 = add_random_valued_impulse_noise(lena_image, density=0.80, random_seed=RANDOM_SEED)
    lena_burst_noise_80, _ = add_structured_burst_noise(lena_image, density=0.80, block_size=16, random_seed=RANDOM_SEED)

    print("Approximate changed pixel ratios for Lena at 30% density")
    print("Salt-and-Pepper:", calculate_changed_pixel_ratio(lena_image, lena_sp_noise))
    print("RVIN:", calculate_changed_pixel_ratio(lena_image, lena_rvin_noise))
    print("Structured Burst:", calculate_changed_pixel_ratio(lena_image, lena_burst_noise))

    show_image_grid(
        images=[lena_image, lena_sp_noise, lena_rvin_noise, lena_burst_noise],
        titles=["Original Lena", "Salt-and-Pepper Noise, 30%", "RVIN, 30%", "Structured Burst Noise, 30%"],
        cols=4,
        figsize=(16, 4),
        save_path=FIGURES_DIR / "lena_30_noise.JPG",
    )

    show_image_grid(
        images=[cameraman_image, cameraman_sp_noise, cameraman_rvin_noise, cameraman_burst_noise],
        titles=["Original Cameraman", "Salt-and-Pepper Noise, 30%", "RVIN, 30%", "Structured Burst Noise, 30%"],
        cols=4,
        figsize=(16, 4),
        save_path=FIGURES_DIR / "camera_30_noise.JPG",
    )

    show_image_grid(
        images=[lena_image, lena_sp_noise_80, lena_rvin_noise_80, lena_burst_noise_80],
        titles=["Original Lena", "Salt-and-Pepper Noise, 80%", "RVIN, 80%", "Structured Burst Noise, 80%"],
        cols=4,
        figsize=(16, 4),
        save_path=FIGURES_DIR / "lena_80_noise.JPG",
    )

    # --------------------------------------------------------
    # Median Filter results
    # --------------------------------------------------------
    lena_sp_median_30 = apply_standard_median_filter(lena_sp_noise, kernel_size=3)
    lena_rvin_median_30 = apply_standard_median_filter(lena_rvin_noise, kernel_size=3)
    lena_burst_median_30 = apply_standard_median_filter(lena_burst_noise, kernel_size=3)

    show_image_grid(
        images=[
            lena_image,
            lena_sp_noise,
            lena_sp_median_30,
            lena_rvin_noise,
            lena_rvin_median_30,
            lena_burst_noise,
            lena_burst_median_30,
        ],
        titles=[
            "Original Lena",
            "S&P Noise, 30%",
            "Median Filter on S&P",
            "RVIN, 30%",
            "Median Filter on RVIN",
            "Burst Noise, 30%",
            "Median Filter on Burst",
        ],
        cols=4,
        figsize=(16, 8),
        save_path=FIGURES_DIR / "8.JPG",
    )

    # --------------------------------------------------------
    # Standard DBRAMF results
    # --------------------------------------------------------
    lena_sp_dbramf_30 = apply_standard_dbramf(lena_sp_noise, max_window_size=5)
    lena_rvin_dbramf_30 = apply_standard_dbramf(lena_rvin_noise, max_window_size=5)
    lena_burst_dbramf_30 = apply_standard_dbramf(lena_burst_noise, max_window_size=5)

    lena_sp_dbramf_80 = apply_standard_dbramf(lena_sp_noise_80, max_window_size=5)
    lena_burst_dbramf_80 = apply_standard_dbramf(lena_burst_noise_80, max_window_size=5)

    show_image_grid(
        images=[
            lena_image,
            lena_sp_noise,
            lena_sp_dbramf_30,
            lena_rvin_noise,
            lena_rvin_dbramf_30,
            lena_burst_noise,
            lena_burst_dbramf_30,
        ],
        titles=[
            "Original Lena",
            "S&P Noise, 30%",
            "DBRAMF on S&P",
            "RVIN, 30%",
            "DBRAMF on RVIN",
            "Burst Noise, 30%",
            "DBRAMF on Burst",
        ],
        cols=4,
        figsize=(16, 8),
        save_path=FIGURES_DIR / "12.JPG",
    )

    # --------------------------------------------------------
    # Proposed GA-DBRAMF results
    # --------------------------------------------------------
    lena_sp_ga_30 = apply_ga_dbramf_balanced(lena_sp_noise, max_window_size=5, detection_mode="binary")
    lena_rvin_ga_30 = apply_ga_dbramf_balanced(lena_rvin_noise, max_window_size=5, detection_mode="rvin", rvin_threshold=45)
    lena_burst_ga_30 = apply_ga_dbramf_balanced(lena_burst_noise, max_window_size=5, detection_mode="binary")

    lena_sp_ga_80 = apply_ga_dbramf_balanced(lena_sp_noise_80, max_window_size=5, detection_mode="binary")
    lena_rvin_ga_80 = apply_ga_dbramf_balanced(lena_rvin_noise_80, max_window_size=5, detection_mode="rvin", rvin_threshold=45)
    lena_burst_ga_80 = apply_ga_dbramf_balanced(lena_burst_noise_80, max_window_size=5, detection_mode="binary")

    show_image_grid(
        images=[
            lena_image,
            lena_sp_noise,
            lena_sp_median_30,
            lena_sp_dbramf_30,
            lena_sp_ga_30,
            lena_burst_noise,
            lena_burst_median_30,
            lena_burst_dbramf_30,
            lena_burst_ga_30,
        ],
        titles=[
            "Original Lena",
            "S&P Noise, 30%",
            "Median on S&P",
            "DBRAMF on S&P",
            "GA-DBRAMF on S&P",
            "Burst Noise, 30%",
            "Median on Burst",
            "DBRAMF on Burst",
            "GA-DBRAMF on Burst",
        ],
        cols=5,
        figsize=(18, 8),
        save_path=FIGURES_DIR / "25.JPG",
    )

    show_image_grid(
        images=[
            lena_image,
            lena_burst_noise_80,
            lena_burst_dbramf_80,
            lena_burst_ga_80,
            lena_sp_noise_80,
            lena_sp_dbramf_80,
            lena_sp_ga_80,
        ],
        titles=[
            "Original Lena",
            "Burst Noise, 80%",
            "DBRAMF on Burst",
            "GA-DBRAMF on Burst",
            "S&P Noise, 80%",
            "DBRAMF on S&P",
            "GA-DBRAMF on S&P",
        ],
        cols=4,
        figsize=(16, 8),
        save_path=FIGURES_DIR / "26.JPG",
    )

    # --------------------------------------------------------
    # Selected metric tests and edge maps
    # --------------------------------------------------------
    selected_metric_results = []

    selected_cases = [
        ("Lena", "Salt-and-Pepper", "30%", "Noisy Image", lena_sp_noise),
        ("Lena", "Salt-and-Pepper", "30%", "Median Filter", lena_sp_median_30),
        ("Lena", "Salt-and-Pepper", "30%", "Standard DBRAMF", lena_sp_dbramf_30),
        ("Lena", "Salt-and-Pepper", "30%", "GA-DBRAMF", lena_sp_ga_30),
        ("Lena", "Structured Burst", "30%", "Noisy Image", lena_burst_noise),
        ("Lena", "Structured Burst", "30%", "Median Filter", lena_burst_median_30),
        ("Lena", "Structured Burst", "30%", "Standard DBRAMF", lena_burst_dbramf_30),
        ("Lena", "Structured Burst", "30%", "GA-DBRAMF", lena_burst_ga_30),
    ]

    for image_name, noise_type, density_label, method_name, restored_image in selected_cases:
        metrics = calculate_all_metrics(lena_image, restored_image)
        selected_metric_results.append({
            "Image": image_name,
            "Noise_Type": noise_type,
            "Density": density_label,
            "Method": method_name,
            "MSE": metrics["MSE"],
            "PSNR": metrics["PSNR"],
            "SSIM": metrics["SSIM"],
            "Pratt_FOM": metrics["Pratt_FOM"],
        })

    selected_metric_df = pd.DataFrame(selected_metric_results)
    selected_metric_df.to_csv(TABLES_DIR / "selected_lena_30_metric_results.csv", index=False)
    print(selected_metric_df)

    original_edges = extract_edge_map(lena_image)
    median_edges = extract_edge_map(lena_sp_median_30)
    dbramf_edges = extract_edge_map(lena_sp_dbramf_30)
    ga_edges = extract_edge_map(lena_sp_ga_30)

    show_image_grid(
        images=[
            original_edges.astype(np.uint8) * 255,
            median_edges.astype(np.uint8) * 255,
            dbramf_edges.astype(np.uint8) * 255,
            ga_edges.astype(np.uint8) * 255,
        ],
        titles=[
            "Original Edge Map",
            "Median Filter Edge Map",
            "DBRAMF Edge Map",
            "GA-DBRAMF Edge Map",
        ],
        cols=4,
        figsize=(16, 4),
        save_path=FIGURES_DIR / "29.JPG",
    )

    # --------------------------------------------------------
    # Full experimental matrix
    # --------------------------------------------------------
    full_experiment_images = {
        "Lena": lena_image,
        "Cameraman": cameraman_image,
    }

    full_noise_types = {
        "salt_and_pepper": "Salt-and-Pepper",
        "rvin": "RVIN",
        "structured_burst": "Structured Burst",
    }

    full_noise_densities = [0.10, 0.30, 0.50, 0.70, 0.80]

    full_methods = {
        "standard_median_filter": "Standard Median Filter",
        "standard_dbramf": "Standard DBRAMF",
        "ga_dbramf": "GA-DBRAMF",
    }

    print("Full experimental matrix is ready.")
    print("Expected restored method results:", 90)
    print("Expected noisy reference results:", 30)
    print("Expected total rows:", 120)

    full_results = []
    experiment_counter = 0
    start_time = time.time()

    for image_index, (image_name, original_image) in enumerate(full_experiment_images.items()):
        for noise_index, (noise_type_key, noise_type_label) in enumerate(full_noise_types.items()):
            for density_index, density in enumerate(full_noise_densities):
                experiment_counter += 1
                current_seed = RANDOM_SEED + image_index * 1000 + noise_index * 100 + density_index * 10
                density_label = f"{int(density * 100)}%"

                print(
                    f"Running experiment {experiment_counter}/30: "
                    f"{image_name}, {noise_type_label}, {density_label}"
                )

                noisy_image = generate_noisy_image(
                    original_image,
                    noise_type=noise_type_key,
                    density=density,
                    random_seed=current_seed,
                )

                noisy_metrics = calculate_all_metrics(original_image, noisy_image)
                full_results.append({
                    "Image": image_name,
                    "Noise_Type": noise_type_label,
                    "Density": density_label,
                    "Density_Value": density,
                    "Method": "Noisy Image",
                    "MSE": noisy_metrics["MSE"],
                    "PSNR": noisy_metrics["PSNR"],
                    "SSIM": noisy_metrics["SSIM"],
                    "Pratt_FOM": noisy_metrics["Pratt_FOM"],
                    "Runtime_Seconds": np.nan,
                })

                for method_key, method_label in full_methods.items():
                    method_start_time = time.time()
                    restored_image = apply_restoration_method(
                        noisy_image,
                        method_name=method_key,
                        noise_type=noise_type_key,
                    )
                    metrics = calculate_all_metrics(original_image, restored_image)
                    method_runtime = time.time() - method_start_time

                    full_results.append({
                        "Image": image_name,
                        "Noise_Type": noise_type_label,
                        "Density": density_label,
                        "Density_Value": density,
                        "Method": method_label,
                        "MSE": metrics["MSE"],
                        "PSNR": metrics["PSNR"],
                        "SSIM": metrics["SSIM"],
                        "Pratt_FOM": metrics["Pratt_FOM"],
                        "Runtime_Seconds": method_runtime,
                    })

    end_time = time.time()

    full_results_df = pd.DataFrame(full_results)
    restored_results_df = full_results_df[full_results_df["Method"] != "Noisy Image"].copy()

    full_results_df.to_csv(RESULTS_DIR / "full_experimental_results.csv", index=False)
    restored_results_df.to_csv(RESULTS_DIR / "restored_method_results_only.csv", index=False)

    print("Full experimental matrix completed.")
    print("Total number of rows:", len(full_results_df))
    print("Restored method rows:", len(restored_results_df))
    print("Total runtime in minutes:", round((end_time - start_time) / 60, 2))

    average_by_method = restored_results_df.groupby("Method")[["MSE", "PSNR", "SSIM", "Pratt_FOM"]].mean().reset_index()
    average_by_method = average_by_method.sort_values(by="PSNR", ascending=False)
    average_by_method.to_csv(TABLES_DIR / "average_by_method.csv", index=False)
    print(average_by_method)

    average_by_noise_and_method = restored_results_df.groupby(["Noise_Type", "Method"])[["MSE", "PSNR", "SSIM", "Pratt_FOM"]].mean().reset_index()
    average_by_noise_and_method.to_csv(TABLES_DIR / "average_by_noise_and_method.csv", index=False)
    print(average_by_noise_and_method)

    psnr_pivot = restored_results_df.pivot_table(
        index=["Noise_Type", "Density"],
        columns="Method",
        values="PSNR",
        aggfunc="mean",
    ).reset_index()
    psnr_pivot.to_csv(TABLES_DIR / "psnr_pivot.csv", index=False)
    print(psnr_pivot)

    ssim_pivot = restored_results_df.pivot_table(
        index=["Noise_Type", "Density"],
        columns="Method",
        values="SSIM",
        aggfunc="mean",
    ).reset_index()
    ssim_pivot.to_csv(TABLES_DIR / "ssim_pivot.csv", index=False)

    fom_pivot = restored_results_df.pivot_table(
        index=["Noise_Type", "Density"],
        columns="Method",
        values="Pratt_FOM",
        aggfunc="mean",
    ).reset_index()
    fom_pivot.to_csv(TABLES_DIR / "fom_pivot.csv", index=False)

    # --------------------------------------------------------
    # Metric curve visualization
    # --------------------------------------------------------
    plot_results_df = restored_results_df.copy()
    plot_results_df["Density_Percent"] = plot_results_df["Density"].str.replace("%", "", regex=False).astype(int)

    plot_metric_curves(plot_results_df, "PSNR", "PSNR (dB)", "39.JPG")
    plot_metric_curves(plot_results_df, "SSIM", "SSIM", "40.JPG")
    plot_metric_curves(plot_results_df, "Pratt_FOM", "Pratt's FOM", "41.JPG")
    plot_average_metric_comparison(average_by_method)

    # --------------------------------------------------------
    # Ablation study
    # --------------------------------------------------------
    lena_burst_dbramf_30_window3 = apply_standard_dbramf(lena_burst_noise, max_window_size=3)
    lena_burst_dbramf_30_window5 = apply_standard_dbramf(lena_burst_noise, max_window_size=5)
    lena_burst_ga_30_window3 = apply_ga_dbramf_balanced(lena_burst_noise, max_window_size=3, detection_mode="binary")
    lena_burst_ga_30_window5 = apply_ga_dbramf_balanced(lena_burst_noise, max_window_size=5, detection_mode="binary")

    window_ablation_results = [
        evaluate_ablation_case(lena_image, lena_burst_dbramf_30_window3, "DBRAMF max window 3x3"),
        evaluate_ablation_case(lena_image, lena_burst_dbramf_30_window5, "DBRAMF max window 5x5"),
        evaluate_ablation_case(lena_image, lena_burst_ga_30_window3, "GA-DBRAMF max window 3x3"),
        evaluate_ablation_case(lena_image, lena_burst_ga_30_window5, "GA-DBRAMF max window 5x5"),
    ]
    window_ablation_df = pd.DataFrame(window_ablation_results)
    window_ablation_df.to_csv(TABLES_DIR / "window_size_ablation_results.csv", index=False)
    print(window_ablation_df)

    lena_burst_80_standard_fallback = apply_standard_dbramf(lena_burst_noise_80, max_window_size=5)
    lena_burst_80_gradient_fallback = apply_ga_dbramf_balanced(lena_burst_noise_80, max_window_size=5, detection_mode="binary")

    fallback_ablation_results = [
        evaluate_ablation_case(lena_image, lena_burst_80_standard_fallback, "Standard DBRAMF fallback"),
        evaluate_ablation_case(lena_image, lena_burst_80_gradient_fallback, "Gradient-aware fallback"),
    ]
    fallback_ablation_df = pd.DataFrame(fallback_ablation_results)
    fallback_ablation_df.to_csv(TABLES_DIR / "fallback_strategy_ablation_results.csv", index=False)
    print(fallback_ablation_df)

    show_image_grid(
        images=[
            lena_image,
            lena_burst_noise,
            lena_burst_dbramf_30_window3,
            lena_burst_dbramf_30_window5,
            lena_burst_ga_30_window3,
            lena_burst_ga_30_window5,
            lena_burst_noise_80,
            lena_burst_80_standard_fallback,
            lena_burst_80_gradient_fallback,
        ],
        titles=[
            "Original Lena",
            "Burst Noise, 30%",
            "DBRAMF 3x3",
            "DBRAMF 5x5",
            "GA-DBRAMF 3x3",
            "GA-DBRAMF 5x5",
            "Burst Noise, 80%",
            "Standard Fallback",
            "Gradient-Aware Fallback",
        ],
        cols=5,
        figsize=(18, 8),
        save_path=FIGURES_DIR / "45.JPG",
    )

    # --------------------------------------------------------
    # Final summary and runtime comparison
    # --------------------------------------------------------
    best_psnr_rows = restored_results_df.loc[
        restored_results_df.groupby(["Image", "Noise_Type", "Density"])["PSNR"].idxmax()
    ].copy()

    best_ssim_rows = restored_results_df.loc[
        restored_results_df.groupby(["Image", "Noise_Type", "Density"])["SSIM"].idxmax()
    ].copy()

    best_fom_rows = restored_results_df.loc[
        restored_results_df.groupby(["Image", "Noise_Type", "Density"])["Pratt_FOM"].idxmax()
    ].copy()

    best_psnr_count = best_psnr_rows["Method"].value_counts().reset_index()
    best_psnr_count.columns = ["Method", "Best_PSNR_Count"]

    best_ssim_count = best_ssim_rows["Method"].value_counts().reset_index()
    best_ssim_count.columns = ["Method", "Best_SSIM_Count"]

    best_fom_count = best_fom_rows["Method"].value_counts().reset_index()
    best_fom_count.columns = ["Method", "Best_FOM_Count"]

    best_method_summary = best_psnr_count.merge(best_ssim_count, on="Method", how="outer")
    best_method_summary = best_method_summary.merge(best_fom_count, on="Method", how="outer")
    best_method_summary = best_method_summary.fillna(0)
    best_method_summary.to_csv(TABLES_DIR / "best_method_summary.csv", index=False)
    print(best_method_summary)

    final_summary_rows = []
    for method_name in average_by_method["Method"]:
        method_subset = restored_results_df[restored_results_df["Method"] == method_name]
        final_summary_rows.append({
            "Method": method_name,
            "Average_MSE": method_subset["MSE"].mean(),
            "Average_PSNR": method_subset["PSNR"].mean(),
            "Average_SSIM": method_subset["SSIM"].mean(),
            "Average_Pratt_FOM": method_subset["Pratt_FOM"].mean(),
            "Average_Runtime_Seconds": method_subset["Runtime_Seconds"].mean(),
        })

    final_summary_df = pd.DataFrame(final_summary_rows)
    final_summary_df.to_csv(TABLES_DIR / "final_summary_with_runtime.csv", index=False)
    print(final_summary_df)

    plot_runtime_comparison(final_summary_df)

    print("All experiments and outputs were generated successfully.")
    print("Figures are saved in:", FIGURES_DIR)
    print("Tables are saved in:", TABLES_DIR)
    print("CSV results are saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
