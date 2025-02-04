# Descriptor_form.py

**Size, Orientation, and Form Descriptors Library**

A lightweight library for computing various descriptors of binary shapes in images using OpenCV, NumPy, SciPy, and scikit-image.  
**Note:** Ensure the object is surrounded by at least 2 black pixels to avoid border artifacts.

---

## Features

### Basic Parameters (`Basic_Parameters` class)
- **`particle_perimeter`**: Compute the object's perimeter.
- **`particle_area`**: Calculate the object's area.
- **`convexhull_area_perimeter`**: Get area and perimeter of the convex hull.
- **`minimum_enclosing_rectangle`**: Determine bounding rectangle dimensions.
- **`equivalent_moment_ellipse`**: Compute equivalent ellipse parameters (orientation, major/minor axes).
- **`maximum_inscribed_circle` / `minimum_circumscribed_circle`**: Find inscribed/circumscribed circles (with optional plotting).
- **`feret_measurments`**: Calculate Feret diameters and statistics.
- **`area_after_erosion_dilation`**: Measure area after morphological operations.
- **`long_axis_orientation`**: Determine orientation via bounding rectangle.
- **`compute_orientation_minimizing_second_moment`**: Compute orientation by minimizing the second moment.

### Form Descriptors (`Form_Descriptors` class)
- **`feret_elongation`**: Ratio of max to min Feret diameter.
- **`diameter_elongation`**: Ratio between circumscribed and inscribed circle radii.
- **`equivalent_area_disc`**: Equivalent disc diameter based on area.
- **`circularity`**: Measure of how close the shape is to a perfect circle.
- **`rectangularity_perimeter` / `rectangularity_area`**: Rectangularity based on perimeter or area.
- **`ellipsoidity_perimeter` / `ellipsoidity_area`**: Compare shape to its equivalent ellipse.
- **`polygon_to_circle_area`**: Ratio comparing particle area with the difference in circle areas.
- **`independant_fourier_descriptors`**: Normalized Fourier descriptors using both radius and contour (XY) methods.

### Fourier Methods (`Fourier` class)
- **`radius_fft`**: Fourier transform of radial distances.
- **`contour_fft`**: Fourier transform of contour coordinates.
- **`get_fft_order`**: Helper to extract a specific Fourier coefficient range.

---

## Usage Example

```python
import cv2
from Descriptor_form import Basic_Parameters, Form_Descriptors

# Load a binary image (object should be white on a black background)
binary_image = cv2.imread('path_to_image.png', cv2.IMREAD_GRAYSCALE)

# Compute basic descriptors
perimeter = Basic_Parameters.particle_perimeter(binary_image)
area = Basic_Parameters.particle_area(binary_image)
orientation = Basic_Parameters.long_axis_orientation(binary_image)

# Compute form descriptors
circularity = Form_Descriptors.circularity(binary_image)
fourier_desc = Form_Descriptors.independant_fourier_descriptors(
    binary_image, k=7, convexhull=True, order1=1, order2=12
)

print("Perimeter:", perimeter)
print("Area:", area)
print("Orientation:", orientation)
print("Circularity:", circularity)
print("Fourier Descriptors:", fourier_desc)
