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
    # img = img.resize((150, 150))

    # Convert the image to an array of RGB values
    ar = np.asarray(img)
    shape = ar.shape
    new_ar = np.array([])
    for row in range(shape[0]):
        for column in range(shape[1]):
            print(ar[row][column])
            print(np.array_equal(ar[row][column], np.array([0, 0, 0])))
            if not (np.array_equal(ar[row][column], np.array([0, 0, 0]))):
                np.append(new_ar, ar[row][column])
    ar = new_ar
    print(ar.shape)
    # ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
  
    # Perform k-means clustering
    NUM_CLUSTERS = 5  # You can adjust this value
    codes, _ = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    # Find the most frequent color
    vecs, _ = scipy.cluster.vq.vq(ar, codes)
    counts, _ = np.histogram(vecs, len(codes))
    index_max = np.argmax(counts)
    dominant_color = codes[index_max]

    return dominant_color

# Example usage
image_path = 'output/alpha/image1.png'
dominant_color = get_dominant_color(image_path)
print(f"The dominant color is RGB: {dominant_color}")

rgb_value = dominant_color
print(convert_rgb_to_name(rgb_value))
