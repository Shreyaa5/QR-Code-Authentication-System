import cv2
import os
import matplotlib.pyplot as plt
import logging
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Fix UnicodeEncodeError: 'charmap' codec can't encode character


# Dataset paths (Update these)
import os

# Use raw strings (r"") or double backslashes (\\) for Windows paths
original_path = "C:/Users/ShreyaPaul/Documents/coding/QR Code Authentication system/QR_Code_Authentication/First_Print"
counterfeit_path = "C:/Users/ShreyaPaul/Documents/coding/QR Code Authentication system/QR_Code_Authentication/Second_Print"

# Ensure the paths exist
if not os.path.exists(original_path):
    print(f"Error: Original QR Code folder not found at {original_path}")

if not os.path.exists(counterfeit_path):
    print(f"Error: Counterfeit QR Code folder not found at {counterfeit_path}")


def load_image(image_path):
    """Load and convert an image to grayscale."""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_samples():
    """Display original and counterfeit QR codes side by side."""
    original_img = load_image(os.path.join(original_path, os.listdir(original_path)[0]))
    counterfeit_img = load_image(os.path.join(counterfeit_path, os.listdir(counterfeit_path)[0]))

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title("Original QR Code")

    plt.subplot(1, 2, 2)
    plt.imshow(counterfeit_img, cmap="gray")
    plt.title("Counterfeit QR Code")

    plt.show()

if __name__ == "__main__":
    show_samples()
