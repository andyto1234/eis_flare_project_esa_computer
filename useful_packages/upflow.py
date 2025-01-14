import numpy as np
from scipy import ndimage
from skimage import filters, morphology

def find_largest_upflow_region(dopp_eis):
    # Assuming dopp_eis is your Doppler velocity map
    data = dopp_eis.data

    # 1. Create a mask to exclude the border region
    border_width = 10  # Adjust this value based on your image
    mask = np.ones_like(data, dtype=bool)
    mask[:border_width, :] = mask[-border_width:, :] = False
    mask[:, :border_width] = mask[:, -border_width:] = False

    # 2. Apply Gaussian blur with edge handling
    smoothed_data = ndimage.gaussian_filter(data, sigma=2, mode='nearest')

    # 3. Apply median filter
    median_filtered = filters.median(smoothed_data, morphology.disk(3))

    # 4. Create mask for upflow areas with a higher threshold, excluding the border
    upflow_mask = (median_filtered < -12) & mask  # Adjust threshold as needed

    # 5. Use morphological operations to clean up the upflow mask
    # Remove small objects from the upflow mask that are smaller than the specified minimum size
    cleaned = morphology.remove_small_objects(upflow_mask, min_size=50)
    # Apply morphological closing to fill small holes in the upflow mask
    cleaned = morphology.closing(cleaned, morphology.disk(3))

    # 6. Label connected regions and find the largest one
    labeled_mask, num_features = ndimage.label(cleaned)
    if num_features > 0:
        sizes = ndimage.sum(cleaned, labeled_mask, range(1, num_features + 1))
        largest_feature_label = sizes.argmax() + 1
        largest_feature_mask = labeled_mask == largest_feature_label
    else:
        largest_feature_mask = np.zeros_like(cleaned)

    # Get the coordinates of the largest upflow region
    if np.any(largest_feature_mask):
        y, x = np.where(largest_feature_mask)
        bottom_left_y, bottom_left_x = np.min(y), np.min(x)
        top_right_y, top_right_x = np.max(y), np.max(x)

        # Convert pixel coordinates to world coordinates and shrink the box by 1 arcsec in x and y
        bottom_left_world = dopp_eis.pixel_to_world((bottom_left_x) * u.pix, (bottom_left_y) * u.pix)
        top_right_world = dopp_eis.pixel_to_world((top_right_x) * u.pix, (top_right_y) * u.pix)
        return bottom_left_world, top_right_world
    else:
        print("No significant upflow region found.")
        return None, None

# Example usage:
# bottom_left_world, top_right_world = find_largest_upflow_region(dopp_eis)
# if bottom_left_world and top_right_world:
#     print(f"World coordinates of the bottom left: {bottom_left_world}")
#     print(f"World coordinates of the top right: {top_right_world}")
# else:
#     print("No significant upflow region found.")