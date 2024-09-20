"""
Size, Orientation and Form Descriptors Librairy
"""

# 1. Librairy used

import math
import cv2
import feret
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import fftpack
from scipy.spatial import ConvexHull, distance
from scipy.ndimage import distance_transform_edt
from skimage import draw, measure
from skimage.morphology import binary_erosion, binary_dilation
from skimage.measure import moments
from skimage.io import imread


# 2. Shape Parameters (used to compute the descriptors)

def particle_perimeter(binary_object):
	"""
	Calculate the perimeter of a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- Perimeter of the object.
	"""
	
	contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea).squeeze()
	
	return cv2.arcLength(contour, True)


def particle_area(binary_object):
	"""
	Calculate the area of a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- Area of the object.
	"""
	
	contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	#contour = contours[0].squeeze()
	contour = max(contours, key=cv2.contourArea).squeeze()
	
	return cv2.contourArea(contour)


def area_after_erosion_dilation(binary_object, kernel_size=3, nb_erosion=1, nb_dilation=1):
	"""
	Calculate the area of a binary object after applying erosion and dilation operations.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	- kernel_size: Size of the erosion/dilation kernel (default is 3x3).
	- nb_erosion: Number of erosion iterations (default is 1).
	- nb_dilation: Number of dilation iterations (default is 1).

	Returns:
	- Area of the object after erosion and dilation.
	"""
	
	kernel = np.ones((kernel_size, kernel_size), np.uint8)

	if nb_erosion >= 0:
		binary_object = cv2.erode(binary_object, kernel, iterations=nb_erosion)

	if nb_dilation >= 0:
		binary_object = cv2.dilate(binary_object, kernel, iterations=nb_dilation)

	contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea).squeeze()
	
	return cv2.contourArea(contour)


def convexhull_area_perimeter(binary_object):
	"""
	Calculate the area and perimeter of the convex hull of a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- Tuple (convex_hull_area, convex_hull_perimeter): where convex_hull_area is the area of the convex hull
	 and convex_hull_perimeter is the perimeter of the convex hull.
	"""

	contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea).squeeze()
	hull = cv2.convexHull(contour)
	
	return cv2.contourArea(hull), cv2.arcLength(hull, True)


# 3. Size Measurments Methods

def bounding_rectangle_area(binary_object):
	"""
	Calculate the parameters of the minimum enclosing rectangle of a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- height: Height of the bounding rectangle.
	- width: Width of the bounding rectangle.
	- area: Area of the bounding rectangle.
	"""
	
	contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea).squeeze()
	_, (height, width), _ = cv2.minAreaRect(contour)
	area = height * width
	
	return height, width, area


def equivalent_moment_ellipse(image):
	"""
	Calculate the parameters of the equivalent moment ellipse of a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- theta: The orientation in degree measured using the second moment of inertia of the ellipse.
	- a: The major axis length.
	- b: The minor axis length.
	"""

	region = measure.regionprops(image)[0]
	# calculate the semi-axes of the equivalent ellipse
	a = region.major_axis_length
	b = region.minor_axis_length
	theta = region.orientation
	
	return np.rad2deg(theta),a,b


def maximum_inscribed_circle(binary_object, display=False):
	"""
	Calculate the maximum inscribed circle in a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	- display: 
 
	Returns:
	- Tuple (maximum_inscribed_radius, coords): where maximum_inscribed_radius is the radius of the circle 
	 and coords is the (row, col) coordinates of the possibles circle's center.
	"""
	
	distance = distance_transform_edt(binary_object)
	maximum_inscribed_radius = np.max(distance)
	coords = np.where(distance == maximum_inscribed_radius)
	# Display the binary with the maximum inscribed circle drawn
	if display == True:
		row, col = draw.circle_perimeter(int(round(coords[0][0])),int(round(coords[1][0])), int(round(maximum_inscribed_radius)))
		fig, ax = plt.subplots(1, 1) 
		img_output = binary_object.copy()
		mask = (row >= 0) & (row < img_output.shape[0]) & (col >= 0) & (col < img_output.shape[1])
		img_output[row[mask], col[mask]] = 1
		# Plotting the center of the circle
		ax.scatter(coords[1], coords[0], color='red') 
		ax.imshow(img_output, cmap="gray")
	
	return maximum_inscribed_radius, coords


def minimum_circumscribed_circle(binary_object, display=False):
	"""
	Computes the minimum circumscribed circle radius.
	
	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	- display: 
	
	Returns:
	- Tuple (radius, center): where radius is the radius of the circle and center is the circle's center coordinates.
	"""
	
	contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea)
	# Find the minimum circumscribed circle using the OpenCV method
	(center, radius) = cv2.minEnclosingCircle(contour)
	# Display the binary with the minimum circumscribed circle drawn
	if display:
		plt.figure(figsize=(6, 6))
		plt.imshow(binary_object, cmap='gray')
		plt.title("Simulated Binary Image with Enclosing Circle")
		plt.axis('off')
		# Draw the found minimum circumscribed circle
		circle_img = np.zeros_like(binary_object)
		cv2.circle(circle_img, (int(center[0]), int(center[1])), int(radius), 255, 1)
		plt.imshow(circle_img, cmap='gray', alpha=0.5)  # Overlay with semi-transparency
		plt.show()
		
	return(radius,center)

#def equivalent_area_disc()
#	return


def max_feret_diameter(binary_object):
	"""
	Calculate the maximum Feret diameter of a binary object.
 
	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- Maximum Feret diameter of the object.
	"""
	
	return feret.max(binary_object)


def mean_feret_diameter(binary_object, angle_step=10):
	"""
	Calculate the mean Feret diameter of a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- Mean Feret diameter of the object.
	- feret_diameters: a list of all the Feret measurments made, the minimum and maximum Feret diameters could be retrieve from this list
	"""
	
	contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea).squeeze()
	feret_diameters = []
	contour = contour.astype(int)
	center = np.mean(contour, axis=0)
	contour = contour - center

	for angle in range(0, 360, angle_step):
		rotated_contour = np.dot(contour, [[math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
										[math.sin(math.radians(angle)), math.cos(math.radians(angle))]])
		rotated_contour = rotated_contour + center
		min_x = np.min(rotated_contour[:, 0])
		max_x = np.max(rotated_contour[:, 0])
		feret_diameters.append(max_x - min_x)

	return np.mean(feret_diameters), feret_diameters


def min_feret_diameter(binary_object):
	"""
	Calculate the minimum Feret diameter of a binary object.

	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).

	Returns:
	- Minimum Feret diameter of the object.
	"""
	
	return feret.min(binary_object)


# 4. Orientation Descriptors

def long_axis_orientation(binary_image):
	"""
	Computes the orientation of a binary shape by fitting a rectangle around it and
	determining the angle between the longest side of the rectangle and the x-axis.

	Parameters:
	- binary_image: A binary image (numpy array) where the shape is white (255) and the background is black (0).

	Returns:
	- angle: The angle of the shape's orientation, adjusted to ensure values fall within a practical range for interpretation.
	"""

	# Find contours in the binary image
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Assuming the largest contour corresponds to the shape of interest
	if contours:
		cnt = max(contours, key=cv2.contourArea)
		# Fit a bounding rectangle around the contour
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		# Compute the dimensions of the rectangle
		edge_lengths = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
		long_side_index = np.argmax(edge_lengths)
		next_index = (long_side_index + 1) % 4
		# Compute the vector representing the longest side
		vector_x, vector_y = box[next_index] - box[long_side_index]
		# Compute the angle with the x-axis
		if (vector_x<=0 and vector_y<=0) or (vector_y<=0 and vector_x>=0) :
			vector_x=-vector_x
			vector_y=-vector_y
		angle = np.arctan2(vector_y, vector_x) * (180.0 / np.pi)
		# Keep the measure between 0 and 180°
		angle = angle % 180
		
		return angle
	else:
		# No countour found
		print("ERROR: No contour found.")
		
		return None


def compute_orientation_minimizing_second_moment(binary_image):
	"""
	Computes the orientation of a binary shape by minimizing the second moment of inertia.
	
	Parameters:
	- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	
	Returns:
	- orientation_minimizing_second_moment: The angle of the shape's second moment axis, adjusted to ensure values fall within a practical range for interpretation.
	"""
	
	# Adjust for the inverted y-axis
	binary_image = np.flipud(binary_image)
	# Calculate moments
	m = moments(binary_image)
	cr = m[0, 1] / m[0, 0]
	cc = m[1, 0] / m[0, 0]
	mu11 = m[1, 1] / m[0, 0] - cr * cc
	mu02 = m[0, 2] / m[0, 0] - cr**2
	mu20 = m[2, 0] / m[0, 0] - cc**2
	# Calculate orientation of the axis that minimizes the second moment
	# This is actually the orientation of the ellipse's major axis
	theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
	orientation = theta * (180 / np.pi)
	# Adjusting to ensure the orientation is within 0 to 180 degrees range
	if orientation < 0:
		orientation += 180
	# To find the orientation of the minor axis (minimizing second moment), we add 90 degrees
	# because the major and minor axes are perpendicular
	orientation_minimizing_second_moment = (orientation + 90) % 180
	
	return orientation_minimizing_second_moment


# 5. Fourier Method

def Rtheta_fft(binary_object, order1, order2, k=7, convexhull=False):
    """
    Computes the Fourier Transform of the radial distances (r) from the center to the contour points in polar coordinates.
    
    Parameters:
    - binary_object: A binary image where the object of interest is white (255) and background is black (0).
    - order1: The starting index of the range of Fourier amplitudes to return.
    - order2: The ending index (inclusive) of the range of Fourier amplitudes to return.
    - k: An integer controlling the number of sample points (2^k). Default is 7.
    - convexhull: A boolean flag to indicate whether to use the convex hull of the contour instead of the contour itself.
    
    Returns:
    - selected_amplitudes: The normalized amplitudes of the Fourier Transform within the [order1,order2] range.
    - selected_transform: The raw Fourier Transform values within the [order1,order2] range.
    """
    
    # Extract region properties from the binary image
    region = measure.regionprops(binary_object)[0]
    
    contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contour = max(contours, key=cv2.contourArea).squeeze()
    
    if convexhull:
        # Compute the convex hull of the contour
        contour_hull = cv2.convexHull(contour).squeeze()
        # Compute the centroid of the convex hull
        center = np.mean(contour_hull, axis=0)
        
        # Plot the original contour, convex hull, and center for visualization
        plt.figure()
        plt.plot(contour[:, 0], contour[:, 1], 'r--', label='Original Contour')
        plt.plot(contour_hull[:, 0], contour_hull[:, 1], 'b-', label='Convex Hull')
        plt.plot(center[0], center[1], 'go', label='Center')
        plt.legend()
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
        plt.title('Convex Hull and Center')
        plt.show()
    else:
        # If not using convex hull, use the original contour and centroid from region properties
        contour_hull = contour
        center = np.array(region.centroid)
    
    x = contour_hull[:, 0] - center[0]
    y = contour_hull[:, 1] - center[1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Sort the contour points by angle (theta)
    sorted_indices = np.argsort(theta)
    r_sorted = r[sorted_indices]
    theta_sorted = theta[sorted_indices]
    
    fr = interpolate.interp1d(theta_sorted, r_sorted, kind='linear', fill_value="extrapolate")
    
    N = 2**k
    
    # Compute equally spaced sample points for Fourier Transform in the theta domain
    sample_points = np.linspace(theta_sorted.min(), theta_sorted.max(), N, endpoint=False)
    
    # Interpolate r at these sample points
    new_r = fr(sample_points)
    
    # Compute the Fourier Transform of the radial distances
    fourier_transform = fftpack.fft(new_r)
    
    # Compute the normalized amplitudes of the Fourier components
    fourier_amplitudes = np.abs(fourier_transform) / N
    
    # Return the amplitude and Fourier transform values for the specified range
    selected_amplitudes = fourier_amplitudes[order1:order2+1]
    selected_transform = fourier_transform[order1:order2+1]
    
    return selected_amplitudes, selected_transform



def XY_fft(binary_object, order1, order2, k=7):
    """
    Computes the Fourier Transform of the contour of an object in a binary image.

    Parameters:
    - binary_object: A binary image where the object of interest is white (255) and the background is black (0).
    - k: An integer controlling the number of sample points (2^k). Default is 7.
    - order1: The starting index of the range of Fourier amplitudes to return.
    - order2: The ending index (inclusive) of the range of Fourier amplitudes to return.

    Returns:
    - amplitudes: A list of Fourier amplitudes within the specified range.
    """
    
    contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).squeeze()
    
    # Check if the contour is a one-dimensional array and reshape it if necessary
    if contour.ndim == 1:
        contour = contour.reshape(-1, 2)
    elif contour.shape[1] != 2:
        raise ValueError("Contour array has an unexpected shape.")
    
    # Compute the cumulative distances along the contour's arc (arc lengths)
    dy = np.diff(contour[:, 1])
    dx = np.diff(contour[:, 0])
    dists = np.cumsum(np.sqrt(dx*dx + dy*dy))
    dists = np.concatenate([[0], dists])  # Add a zero at the start for correct length alignment
    
    # Perform linear interpolation along the contour based on the arc lengths
    fx = interpolate.interp1d(dists, contour[:, 1], kind='linear')
    fy = interpolate.interp1d(dists, contour[:, 0], kind='linear')
    
    # Compute equally spaced sample points for the Fourier Transform
    max_dist = dists[-1]
    sample_points = np.linspace(0, max_dist, 2**k, endpoint=False)
    
    # Interpolate the contour at the sample points
    new_contour = np.column_stack([fx(sample_points), fy(sample_points)])
    
    # Convert the contour coordinates to complex numbers (for Fourier Transform)
    complex_contour = new_contour[:, 0] + 1j * new_contour[:, 1]
    
    # Compute the Fourier Transform of the complex contour
    fourier_transform = fftpack.fft(complex_contour)
    
    amplitudes = np.abs(fourier_transform)
    
    return amplitudes[order1: order2 + 1]

