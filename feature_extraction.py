import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os

def extract_lbp_features(image):
    """Compute Local Binary Pattern (LBP) features and return a fixed-size feature vector."""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)

    # Ensure LBP features have a fixed size (e.g., 30 bins)
    fixed_size = 30
    if len(hist) > fixed_size:
        hist = hist[:fixed_size]  # Truncate
    else:
        hist = np.pad(hist, (0, fixed_size - len(hist)))  # Pad with zeros

    return hist

def extract_edge_features(image):
    """Compute edge features using Canny Edge Detection and return a fixed-size feature vector."""
    edges = cv2.Canny(image, 100, 200)

    # Flatten the edges and set a fixed size
    edge_features = edges.flatten()
    fixed_size = 500  # Define fixed size for edge features

    if len(edge_features) > fixed_size:
        edge_features = edge_features[:fixed_size]  # Truncate
    else:
        edge_features = np.pad(edge_features, (0, fixed_size - len(edge_features)))  # Pad with zeros

    return edge_features

def extract_features(image):
    """Combine multiple feature extraction techniques and ensure consistent feature length."""
    lbp_features = extract_lbp_features(image)
    edge_features = extract_edge_features(image)

    # Ensure all feature vectors are concatenated correctly
    final_features = np.hstack([lbp_features, edge_features])

    return final_features

if __name__ == "__main__":
    from data_preprocessing import load_image, original_path

    # Ensure there's at least one image in the dataset
    image_files = os.listdir(original_path)
    if len(image_files) == 0:
        print(f"No images found in {original_path}. Please add images and try again.")
    else:
        sample_image = load_image(os.path.join(original_path, image_files[0]))
        if sample_image is not None:
            features = extract_features(sample_image)
            print("Extracted Feature Vector Shape:", features.shape)
        else:
            print("Error: Could not load image.")
