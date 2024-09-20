## Readme : Descriptor_form.py

**Size, Orientation and Form Descriptors Library**

This library provides a collection of functions for computing various size, orientation, and form descriptors of binary shapes in images. It leverages libraries like OpenCV, SciPy, and NumPy for image processing and mathematical operations.

**1. Library Imports**

The library imports necessary libraries for image processing, numerical computations, and data manipulation.


* OpenCV (cv2)
* NumPy 
* SciPy 
* Skimage 

(Optional)

* Pandas 
* Matplotlib 
* Feret 


**2. Shape Parameters**

This section defines functions to calculate various shape parameters based on a binary image containing the object of interest.

* `particle_perimeter(binary_object)`: Calculates the perimeter of the object.
* `particle_area(binary_object)`: Calculates the area of the object.
* `area_after_erosion_dilation(binary_object, kernel_size=3, nb_erosion=1, nb_dilation=1)`: Calculates the area after applying erosion and dilation operations.
* `convexhull_area_perimeter(binary_object)`: Calculates the area and perimeter of the convex hull of the object.

**3. Size Measurement Methods**

This section provides functions for computing different size measurements of the object.

* `bounding_rectangle_area(binary_object)`: Calculates the height, width, and area of the minimum bounding rectangle.
* `equivalent_moment_ellipse(image)`: Calculates the parameters (theta, major axis length, minor axis length) of the equivalent moment ellipse.
* `maximum_inscribed_circle(binary_object, display=False)`: Calculates the radius and coordinates of the maximum inscribed circle. (Optional display with circle drawn on the image)
* `minimum_circumscribed_circle(binary_object, display=False)`: Calculates the radius and center of the minimum circumscribed circle. (Optional display with circle drawn on the image)
* `max_feret_diameter(binary_object)`: Calculates the maximum Feret diameter of the object.
* `mean_feret_diameter(binary_object, angle_step=10)`: Calculates the mean Feret diameter and returns a list of all Feret diameter measurements.
* `min_feret_diameter(binary_object)`: Calculates the minimum Feret diameter of the object.

**4. Orientation Descriptors**

This section defines functions for determining the orientation of the object.

* `long_axis_orientation(binary_image)`: Calculates the orientation of the shape based on the longest side of the bounding rectangle.
* `compute_orientation_minimizing_second_moment(binary_image)`: Calculates the orientation that minimizes the second moment of inertia (major axis of the ellipse).

**5. Fourier Method**

This section provides functions for computing the Fourier Transform of the object's shape.

* `Rtheta_fft(binary_object, order1, order2, k=7, convexhull=False)`: Computes the Fourier Transform of radial distances from the center to contour points in polar coordinates. It allows selecting a range of Fourier amplitudes to return and offers the option to use the convex hull instead of the original contour.
* `XY_fft(binary_object, order1, order2, k=7)`: Computes the Fourier Transform of the object's contour coordinates. It allows selecting a range of Fourier amplitudes to return.

