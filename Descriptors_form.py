"""
Size, Orientation and Form Descriptors Librairy

IMPORTANT INFORMATION:
For the binary objects (binary images), the border of the object of interest must not touch the edge of the image.
It is necessary to add a border of at least 2 black pixels around the object of interest.
"""
################################### A. Python Librairies ######################################################
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
###############################################################################################################



################################### B. Basic Object Parameters ################################################
class Basic_Parameters:
	def __init__(self):
		pass

	## B.1 Basic object characteristics
	@staticmethod
	def particle_perimeter(binary_object):
		"""
		Calculates the perimeter of a binary object.
	
		Parameters:
		- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	
		Returns:
		- Perimeter of the object.
		"""
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours, key=cv2.contourArea).squeeze()
		return cv2.arcLength(contour, True)
		
	@staticmethod
	def particle_area(binary_object):
		"""
		Calculates the area of a binary object.
	
		Parameters:
		- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	
		Returns:
		- Area of the object.
		"""
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours, key=cv2.contourArea).squeeze()
		return cv2.contourArea(contour)

	@staticmethod
	def convexhull_area_perimeter(binary_object):
		"""
		Calculates the area and perimeter of the convex hull of a binary object.
	
		Parameters:
		- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	
		Returns:
		- convex_hull_area: the area of the convex hull
		- convex_hull_perimeter: the perimeter of the convex hull.
		"""
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours, key=cv2.contourArea).squeeze()
		hull = cv2.convexHull(contour)
		convex_hull_area = cv2.contourArea(hull)
		convex_hull_perimeter = cv2.arcLength(hull, True)
		return convex_hull_area, convex_hull_perimeter

	## B.2 Size measurments methods
	@staticmethod
	def maximum_inscribed_circle(binary_object, display_plot=False):
		"""
		Calculates the maximum inscribed circle in a binary object.
	
		Parameters:
		- binary_object: A binary image where the object of interest is white (255) and background is black (0).
		- display_plot (bool): if true a matplolib plot showing the measurment is displayed.
	 
		Returns:
		- maximum_inscribed_radius (float): the radius of the inscribed circle
		- coords (row, col): coordinates of the possibles circle's center.
		"""
		distance = distance_transform_edt(binary_object)
		maximum_inscribed_radius = np.max(distance)
		coords = np.where(distance == maximum_inscribed_radius)
		# Display the binary with the maximum inscribed circle drawn
		if display_plot == True:
			row, col = draw.circle_perimeter(int(round(coords[0][0])),int(round(coords[1][0])), int(round(maximum_inscribed_radius)))
			fig, ax = plt.subplots(1, 1) 
			img_output = binary_object.copy()
			mask = (row >= 0) & (row < img_output.shape[0]) & (col >= 0) & (col < img_output.shape[1])
			img_output[row[mask], col[mask]] = 1
			# Plotting the center of the circle
			ax.scatter(coords[1], coords[0], color='red') 
			ax.imshow(img_output, cmap="gray")
		return maximum_inscribed_radius, coords

	@staticmethod
	def minimum_circumscribed_circle(binary_object, display_plot=False):
		"""
		Computes the minimum circumscribed circle radius.
		
		Parameters:
		- binary_object: A binary image where the object of interest is white (255) and background is black (0).
		- display_plot (bool): if true a matplolib plot showing the measurment is displayed. 
		
		Returns:
		- radius (float): the minimum circumscribed circle's radius.
		- center (row, col): minimum circumscribed circle's center coordinates.
		"""
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours, key=cv2.contourArea)
		# Find the minimum circumscribed circle using the OpenCV method
		(center, radius) = cv2.minEnclosingCircle(contour)
		# Display the binary with the minimum circumscribed circle drawn
		if display_plot == True:
			plt.figure(figsize=(6, 6))
			plt.imshow(binary_object, cmap='gray')
			plt.title("Simulated Binary Image with Enclosing Circle")
			plt.axis('off')
			# Draw the found minimum circumscribed circle
			circle_img = np.zeros_like(binary_object)
			cv2.circle(circle_img, (int(center[0]), int(center[1])), int(radius), 255, 1)
			plt.imshow(circle_img, cmap='gray', alpha=0.5)  # Overlay with semi-transparency
			plt.show()
		return radius, center

	@staticmethod
	def feret_measurments(binary_object, angle_step=5):
		"""
		Computes the maximum, minimum, mean and relative standard deviation of the Feret diameters of a binary object.
		
		Parameters:
		- binary_object: A binary image where the object of interest is white (255) and background is black (0).
		- angle_step: The step size for angle rotation in degrees (default: 5).
		
		Returns:
		- max_feret: Maximum Feret diameter.
		- min_feret: Minimum Feret diameter.
		- mean_feret: Mean Feret diameter of the object.
		- RSD_feret: Relative standard deviation (RSD) of the Feret diameters (in %).
		- feret_diameters: A list of all the Feret measurements made.
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
		# Calculate required values
		max_feret = np.max(feret_diameters)
		min_feret = np.min(feret_diameters)
		mean_feret = np.mean(feret_diameters)
		RSD_feret = (np.std(feret_diameters, ddof=1) / mean_feret * 100)
		return max_feret, min_feret, mean_feret, RSD_feret, feret_diameters

	@staticmethod
	def minimum_enclosing_rectangle(binary_object):
		"""
		Calculates the parameters of the minimum enclosing rectangle of a binary object.
	
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

	@staticmethod
	def area_after_erosion_dilation(image, kernel_size=3, nb_erosion=1, nb_dilation=1):
		"""
		Calculates the area of a binary object after applying erosion and dilation operations.
		
		Parameters:
		- image: A binary image where the object of interest is white (255) and background is black (0).
		- kernel_size: Size of the erosion/dilation kernel (default is 3x3).
		- nb_erosion: Number of erosion iterations (default is 1).
		- nb_dilation: Number of dilation iterations (default is 1).
		
		Returns:
		- Area of the object after erosion and dilation.
		"""
		binary_object = image.copy()
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		
		if nb_erosion >= 0:
			binary_object = cv2.erode(binary_object, kernel, iterations=nb_erosion)
		if nb_dilation >= 0:
			binary_object = cv2.dilate(binary_object, kernel, iterations=nb_dilation)
		
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours, key=cv2.contourArea).squeeze()
		return cv2.contourArea(contour)

	@staticmethod
	def equivalent_moment_ellipse(binary_object):
		"""
		Calculates the parameters of the equivalent moment ellipse of a binary object.
	
		Parameters:
		- binary_object: A binary image where the object of interest is white (255) and background is black (0).
	
		Returns:
		- theta: The orientation in degree measured using the second moment of inertia of the ellipse.
		- a: The major axis length.
		- b: The minor axis length.
		"""
	
		region = measure.regionprops(binary_object)[0]
		# calculate the semi-axes of the equivalent ellipse
		a = region.major_axis_length
		b = region.minor_axis_length
		theta = region.orientation
		return np.rad2deg(theta), a, b

	## B.3 Orientation measurments
	@staticmethod
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
			if (vector_x<=0 and vector_y<=0) or (vector_y<=0 and vector_x>=0):
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

	@staticmethod
	def compute_orientation_minimizing_second_moment(binary_image):
		"""
		Computes the orientation of a binary shape by minimizing the second moment of inertia.
		
		Parameters:
		- binary_image: A binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- orientation_minimizing_second_moment: The angle of the shape's second moment axis, adjusted to ensure values fall within a practical range for 
		  interpretation.
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
###############################################################################################################



################################### C. Form Descriptors Definitions ###########################################
class Form_Descriptors:
	def __init__(self):
		pass
	
	@staticmethod
	def feret_elongation(binary_object):
		"""
		Computes the elongation of a shape using the Feret diameters.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The ratio of the maximum Feret diameter to the minimum Feret diameter.
		"""
		max_feret, min_feret, _, _, _ = Basic_Parameters.feret_measurments(binary_object)
		return max_feret / min_feret
	
	@staticmethod
	def diameter_elongation(binary_object):
		"""
		Computes the elongation ratio between the minimum circumscribed circle and the maximum inscribed circle radius.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The ratio of the minimum circumscribed circle radius to the maximum inscribed circle radius.
		"""
		min_circ_radius, _ = Basic_Parameters.minimum_circumscribed_circle(binary_object)
		max_inscribed_radius, _ = Basic_Parameters.maximum_inscribed_circle(binary_object)
		return min_circ_radius / max_inscribed_radius
	
	@staticmethod
	def equivalent_area_disc(binary_object):
		"""
		Computes the equivalent diameter of a disc with the same area as the shape.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The equivalent diameter of the shape based on its area.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		return np.sqrt(4 * particle_area / np.pi)
	
	@staticmethod
	def circularity(binary_object):
		"""
		Computes the circularity of a shape, measuring how close it is to a perfect circle.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The circularity value, with 1 being a perfect circle.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		particle_perimeter = Basic_Parameters.particle_perimeter(binary_object)
		return 4 * np.pi * particle_area / (particle_perimeter ** 2)
	
	@staticmethod
	def rectangularity_perimeter(binary_object):
		"""
		Computes the rectangularity of a shape based on its perimeter.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The ratio of the perimeter to the enclosing rectangle's perimeter.
		"""
		particle_perimeter = Basic_Parameters.particle_perimeter(binary_object)
		height, width, _ = Basic_Parameters.minimum_enclosing_rectangle(binary_object)
		return particle_perimeter / (2 * height + 2 * width)
	
	@staticmethod
	def rectangularity_area(binary_object):
		"""
		Computes the rectangularity of a shape based on its area.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The ratio of the particle area to the enclosing rectangle's area.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		_, _, area = Basic_Parameters.minimum_enclosing_rectangle(binary_object)
		return particle_area / area
	
	@staticmethod
	def ellipsoidity_perimeter(binary_object):
		"""
		Computes the ellipsoidity of a shape based on its perimeter. The equivalent ellipse perimeter is computed using the Ramanujan formula.
		It is the best approximation using simple mathematical operators and parameters.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The ratio of the shape's perimeter to the approximated ellipse perimeter.
		"""
		particle_perimeter = Basic_Parameters.particle_perimeter(binary_object)
		theta, a, b = Basic_Parameters.equivalent_moment_ellipse(binary_object)
		ellipse_perimeter = np.pi * (1.5 * (a + b) - np.sqrt((1.5 * a + 0.5 * b) * (0.5 * a + 1.5 * b)))
		return particle_perimeter / ellipse_perimeter
	
	@staticmethod
	def ellipsoidity_area(binary_object):
		"""
		Computes the ellipsoidity of a shape based on its area.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The ratio of the shape's area to the equivalent moment ellipse area.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		_, a, b = Basic_Parameters.equivalent_moment_ellipse(binary_object)
		ellipse_area = np.pi * a * b / 4
		return particle_area / ellipse_area
	
	@staticmethod
	def polygon_to_circle_area(binary_object):
		"""
		Computes the ratio between the shape's area and the difference of the minimum circumscribed circle area and the maximum inscribed circle area. 
		This formula has been designed for this paper. Its role is to describe the difference in area between the minimum circumscribed circle
		and the maximum inscribed circle normalized to the particle area. This difference decreases with a higher number of sides for ideal shapes.
		
		Parameters:
		- binary_object: A binary image representing the shape.
		
		Returns:
		- The computed ratio.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		min_circ_radius, _ = Basic_Parameters.minimum_circumscribed_circle(binary_object)
		max_inscribed_radius, _ = Basic_Parameters.maximum_inscribed_circle(binary_object)
		return particle_area / ((np.pi * min_circ_radius ** 2) - (np.pi * max_inscribed_radius ** 2))
	
	@staticmethod
	def independant_fourier_descriptors(binary_object, k=7, convexhull=False, order1=1, order2=12):
		"""
		Computes the normalized fourier amplitudes as independant descriptors.
		
		Parameters:
		- binary_object (ndarray): a binary image representing the shape.
		- k (int): determines the number of Fourier coefficients (N = 2^k).
		- convexhull (bool): whether to use the convex hull of the contour.
		- order1 (int): the first harmonic needed.
		- order2 (int): the last harmonic needed.
	
		Returns:		
		- Rtheta (list): list of the amplitudes beetween order 1 and 2 for the particle using the Rtheta method.
		- Rtheta_ch (list): list of the amplitudes beetween order 1 and 2 for the convex hull using the Rtheta method.
		- xy (list): list of the amplitudes beetween order 1 and 2 for the particle using the XY method.
		- xy_ch (list): list of the amplitudes beetween order 1 and 2 for the convex hull using the XY method.
		"""
		# Compute Fourier amplitudes using the Rtheta method (radius FFT)
		Rtheta = Fourier.get_fft_order(Fourier.radius_fft(binary_object, k, convexhull),order1,order2)
		# Compute Fourier amplitudes using the XY method (contour FFT)
		xy = Fourier.get_fft_order(Fourier.contour_fft(binary_object, k, convexhull),order1,order2)
		# If convex hull is used, compute the Fourier descriptors for the convex hull
		if convexhull == True:
			# Compute Fourier amplitudes using the Rtheta method (radius FFT)
			Rtheta_ch = Fourier.get_fft_order(Fourier.radius_fft(binary_object, k, convexhull=True),order1,order2)
			# Compute Fourier amplitudes using the XY method (contour FFT)
			xy_ch = Fourier.get_fft_order(Fourier.contour_fft(binary_object, k, convexhull=True),order1,order2)
		else:
			Rtheta_ch = None
			xy_ch = None
		return Rtheta, Rtheta_ch, xy, xy_ch
###############################################################################################################



################################### F. Rtheta and Ellipctic Fourier Definitions ###############################
class Fourier:
	def __init__(self):
		pass
	
	@staticmethod
	def radius_fft(image, k=7, convexhull=False, return_full_spectrum=False, display_plot=False):
		"""
		Computes the Fourier Transform of the radial distances of a particle's contour (Rtheta method).
	
		Parameters:
		- image (ndarray): binary image of the particle.
		- k (int): determines the number of Fourier coefficients (N = 2^k).
		- convexhull (bool): whether to use the convex hull of the contour.
		- return_full_spectrum (bool): whether to return the full Fourier spectrum.
		- display_plot (bool): if true a matplolib plot showing the convex hull is displayed.
	
		Returns:
		- ndarray: fourier amplitudes (and optionally the full spectrum).
		"""
		# Compute region properties
		regions = measure.regionprops(image)
		if not regions:
			raise ValueError("No regions found in the image.")
		region = regions[0]
		# Find contours
		binary_object = image.copy()
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not contours:
			raise ValueError("No contours found in the image.")
		# Select the largest contour based on area
		largest_contour = max(contours, key=cv2.contourArea)
		# Reshape contour to (n_points, 2)
		contour = largest_contour.squeeze()
		if convexhull:
			# Compute convex hull
			contour_hull = cv2.convexHull(largest_contour).squeeze()
			if contour_hull.ndim != 2 or contour_hull.shape[1] != 2:
				raise ValueError("Convex hull has an unexpected shape after squeezing.")
			# Calculate centroid using image moments
			moments = cv2.moments(contour_hull)
			if moments["m00"] == 0:
				raise ValueError("Zero division error while calculating centroid from moments.")
			center_x = moments["m10"] / moments["m00"]
			center_y = moments["m01"] / moments["m00"]
			center = np.array([center_x, center_y])
		else:
			# Compute centroid using regionprops (region.centroid returns (row, col) -> (y, x))
			center = np.array([region.centroid[1], region.centroid[0]])
		#  Visualization of Contours and Center
		if display_plot == True:
			print(f"Computed center: x = {center[0]}, y = {center[1]}")
			plt.figure()
			plt.plot(contour[:, 0], contour[:, 1], 'r--', label='Original Contour')
			plt.plot(contour_hull[:, 0], contour_hull[:, 1], 'b-', label='Convex Hull')
			plt.plot(center[0], center[1], 'go', label='Center')
			plt.legend()
			plt.gca().invert_yaxis()
			plt.title('Convex Hull and Center')
			plt.xlabel('X')
			plt.ylabel('Y')
			plt.show()
		# Compute relative coordinates
		contour_hull_rel = contour_hull if convexhull else contour
		x = contour_hull_rel[:, 0] - center[0]
		y = contour_hull_rel[:, 1] - center[1]
		# Visualization of Relative Contour
		if display_plot == True:
			# Check if center is correctly subtracted
			print(f"Relative coordinates: x range = [{x.min()}, {x.max()}], y range = [{y.min()}, {y.max()}]")
			plt.figure()
			plt.plot(x, y, 'b-', label='Contour Relative to Center')
			plt.plot(0, 0, 'go', label='Center (0,0)')
			plt.legend()
			plt.gca().invert_yaxis()
			plt.title('Contour Relative to Center')
			plt.xlabel('Relative X')
			plt.ylabel('Relative Y')
			plt.axis('equal')  # Ensure equal scaling
			plt.show()
		# Compute polar coordinates
		r = np.sqrt(x**2 + y**2)
		theta = np.arctan2(y, x)
		# Sort by theta
		sorted_indices = np.argsort(theta)
		r_sorted = r[sorted_indices]
		theta_sorted = theta[sorted_indices]
		# Handle angle wrapping for interpolation
		theta_sorted = np.unwrap(theta_sorted)
		theta_sorted = np.mod(theta_sorted, 2 * np.pi)
		# Interpolation
		fr = interpolate.interp1d(theta_sorted, r_sorted, kind='linear', fill_value="extrapolate")
		N = 2**k
		sample_points = np.linspace(0, 2 * np.pi, N, endpoint=False)
		new_r = fr(sample_points)
		# Fourier transform
		fourier_transform = fftpack.fft(new_r)
		fourier_amplitudes = np.abs(fourier_transform) / N
		# Optionally plot the Fourier Spectrum
		if return_full_spectrum:
			spectrum = fourier_transform[:N//2]
			return fourier_amplitudes[:N//2], spectrum
		else:
			return fourier_amplitudes[:N//2]
	
	@staticmethod
	def contour_fft(image, k=7, convexhull=False, display_plot=False):
		"""
		Computes the Fourier Transform of the contour of an object in a binary image using x and y positions (XY method).
		
		Parameters:
		- image (ndarray): binary image of the particle.
		- k (int): determines the number of Fourier coefficients (N = 2^k).
		- convexhull (bool): whether to use the convex hull of the contour.
		- display_plot (bool): if true a matplolib plot showing the convex hull is displayed.
		
		Returns:
		- ndarray: fourier amplitudes (and optionally the full spectrum).
		"""
		binary_object=image.copy()
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours, key=cv2.contourArea).squeeze()
		# Check if the contour is two-dimensional and handle accordingly
		if convexhull:
			contour_hull = cv2.convexHull(contour).squeeze()
			center = np.mean(contour_hull, axis=0)
			contour = contour_hull
			# Plotting the original contour, convex hull, and center
			if display_plot == True:
				plt.figure()
				plt.plot(contour[:, 0], contour[:, 1], 'r--', label='Original Contour')
				plt.plot(contour_hull[:, 0], contour_hull[:, 1], 'b-', label='Convex Hull')
				plt.plot(center[0], center[1], 'go', label='Center')
				plt.legend()
				plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
				plt.title('Convex Hull and Center')
				plt.show()
	
		if contour.ndim == 1:
			# Reshape contour to a 2D array of shape (n, 2)
			contour = contour.reshape(-1, 2)
		elif contour.shape[1] != 2:
			# Handle other unexpected shapes or raise an error
			raise ValueError("Contour array has an unexpected shape.")
		# Compute arc lengths
		dy = np.diff(contour[:, 1])
		dx = np.diff(contour[:, 0])
		dists = np.cumsum(np.sqrt(dx*dx + dy*dy))
		dists = np.concatenate([[0], dists])
		# Linear interpolation for contour
		fx = interpolate.interp1d(dists, contour[:, 1], kind='linear')
		fy = interpolate.interp1d(dists, contour[:, 0], kind='linear')
		# Compute equally spaced points for Fourier Transform
		max_dist = dists[-1]
		N=2**k
		sample_points = np.linspace(0, max_dist,N, endpoint=False)
		new_contour = np.column_stack([fx(sample_points), fy(sample_points)])
		# Convert the contour coordinates to complex numbers
		complex_contour = new_contour[:, 0] + 1j * new_contour[:, 1]
		# Compute Fourier transform on the contour
		fourier_transform = fftpack.fft(complex_contour)
		amplitudes = np.abs(fourier_transform) / N
		return amplitudes[:int(N/2)]
	
	@staticmethod
	def get_fft_order(amplitudes,order1,order2):
		return amplitudes[order1 : order2+1]
###############################################################################################################
