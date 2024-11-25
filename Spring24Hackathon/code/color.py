from PIL import Image
import numpy as np
import scipy.cluster
from webcolors import rgb_to_name


def convert_rgb_to_name(rgb_tuple):
    try:
        named_color = rgb_to_name(rgb_tuple, spec='css3')
        return f"The closest color name is: {named_color}"
    except ValueError:
        return "No defined color name found for this RGB value."
   
def get_dominant_color(image_path):
    # Open the image
    img = Image.open(image_path)

    # Resize the image (optional, for faster processing)
    #img = img.resize((150, 150))

    # Convert the image to an array of RGB values
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
   
   
    # Filter out white pixels (assuming RGB value for white is [255, 255, 255])
    non_white_indices = np.any(ar > 150 , axis=1)
    ar_non_white = ar[non_white_indices]
   
    np.average(ar_non_white, axis=0)

    if len(ar_non_white) == 0:
        # If there are no non-white pixels, return None
        return None


    # Perform k-means clustering
    NUM_CLUSTERS = 12  # You can adjust this value
    codes, _ = scipy.cluster.vq.kmeans(ar_non_white, NUM_CLUSTERS)

    # Find the most frequent color
    vecs, _ = scipy.cluster.vq.vq(ar_non_white, codes)
    counts, _ = np.histogram(vecs, len(codes))
    index_max = np.argmax(counts)
    dominant_color = codes[index_max]

    return dominant_color

# Example usage
image_path = 'output/alpha/blue_shirt.png'
dominant_color = np.round(get_dominant_color(image_path))
print(f"The dominant color is RGB: {dominant_color}")

rgb_value = dominant_color
print(convert_rgb_to_name(np.round(rgb_value)))

