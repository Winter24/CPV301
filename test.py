import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_region(image, threshold):
    """ Recursively split the image into quadrants based on variance threshold. """
    mean_var = np.var(image)
    if mean_var > threshold:
        h, w = image.shape[:2]
        if h < 2 or w < 2:
            return [image]
        return (split_region(image[:h // 2, :w // 2], threshold) +
                split_region(image[:h // 2, w // 2:], threshold) +
                split_region(image[h // 2:, :w // 2], threshold) +
                split_region(image[h // 2:, w // 2:], threshold))
    else:
        return [image]

def merge_regions(regions):
    """ Merge regions based on the similarity of mean color values """
    i = 0
    while i < len(regions) - 1:
        mean1 = np.mean(regions[i], axis=(0, 1))
        j = i + 1
        while j < len(regions):
            mean2 = np.mean(regions[j], axis=(0, 1))
            if np.linalg.norm(mean1 - mean2) < 15:  # Color difference threshold, adjust as needed
                regions[i] = cv2.addWeighted(regions[i], 0.5, regions[j], 0.5, 0)
                del regions[j]
            else:
                j += 1
        i += 1
    return regions

def plot_segments(segments):
    """ Plot each segment found """
    plt.figure(figsize=(10, 10))
    for i, segment in enumerate(segments):
        plt.subplot(int(np.ceil(np.sqrt(len(segments)))), int(np.ceil(np.sqrt(len(segments)))), i + 1)
        plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

# Load image
image_path = './cat.jpeg'  # Modify this with your image path
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print("Error loading image")
else:
    # Split image based on variance threshold
    variance_threshold = 1000  # Modify this threshold as per your image characteristics
    segments = split_region(image, variance_threshold)

    # Merge segments
    merged_segments = merge_regions(segments)

    # Display results
    plot_segments(merged_segments)
