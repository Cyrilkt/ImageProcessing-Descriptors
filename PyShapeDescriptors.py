"""
Size, Orientation, Form, Roundness and Roughness Descriptors Librairy

IMPORTANT INFORMATION:
For the binary objects (binary images), the border of the object of interest must not touch the edge of the image.
It is necessary to add a border of at least 2 black pixels around the object of interest.
"""
################################### A. Python Librairies ######################################################
import math
import cv2
import pywt
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull, distance
from scipy.spatial.distance import euclidean
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import BSpline, splprep, splev, interp1d
from scipy.optimize import minimize_scalar
from scipy import interpolate
from scipy import fftpack
from scipy.fft import fft2

from skimage import io
from skimage import draw, measure
from skimage.morphology import binary_erosion, binary_dilation
from skimage.measure import moments
from sklearn.decomposition import PCA
from skimage.io import imread

from shapely.geometry import LineString

import matplotlib.pyplot as plt
from tqdm import tqdm
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
	
		Returns:
		- convex_hull_area: the area of the convex hull.
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
	def equivalent_area_disc(binary_object):
		"""
		Computes the equivalent diameter of a disc with the same area as the shape.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The equivalent diameter of the shape based on its area.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		return np.sqrt(4 * particle_area / np.pi)
	
	@staticmethod
	def maximum_inscribed_circle(binary_object, display_plot=False):
		"""
		Calculates the maximum inscribed circle radius in a binary object.
	
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- display_plot (bool): if true a matplolib plot showing the measurment is displayed.
	 
		Returns:
		- maximum_inscribed_radius (float): the radius of the circle.
		- coords (row, col): the  coordinates of the possibles circle's center. The first array is the y coordinates, the second the x.
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- display_plot (bool): if true a matplolib plot showing the measurment is displayed. 
		
		Returns:
		- radius (float): the radius of the minimum circumscribed circle.
		- center (x, y): the minimum circumscribed circle's center coordinates.
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
			plt.scatter(x=int(center[0]), y=int(center[1]))
			plt.show()
		return radius, center
	
	@staticmethod
	def feret_measurments(binary_object, angle_step=2):
		"""
		Computes the maximum, minimum, mean and relative standard deviation of the Feret diameters of a binary object.
	
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- angle_step (int): the step size for angle rotation in degrees (default: 5).
	
		Returns:
		- max_feret: maximum Feret diameter.
		- min_feret: minimum Feret diameter.
		- mean_feret: mean Feret diameter of the object.
		- RSD_feret: relative standard deviation (RSD) of the Feret diameters (in %).
		- feret_diameters: a list of all the Feret measurements made.
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
	
		Returns:
		- height: the height of the minimum enclosing rectangle.
		- width: the width of the minimum enclosing rectangle.
		- area: the area of the minimum enclosing rectangle.
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
		- image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- kernel_size (int): the size of the erosion/dilation kernel (default is 3x3).
		- nb_erosion (int): number of erosion iterations (default is 1).
		- nb_dilation (int): number of dilation iterations (default is 1).
	
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
	def equivalent_moment_ellipse(binary_object, display_plot=False):
		"""
		Calculates the parameters of the equivalent moment ellipse of a binary object.
	
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- display_plot (bool): if true a matplolib plot showing the equivalent moment ellipse is displayed.
	
		Returns:
		- a (float): the major axis length.
		- b (float): the minor axis length.
		- theta (float): the orientation in degree measured using the second moment of inertia of the ellipse.
		- perimeter (float): perimeter of the equivalent moment ellipse.
		- area (float): area of the equivalent moment ellipse.
		"""
		# Get contours of the object
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours,key=cv2.contourArea).squeeze()
		# Fit ellipse to the contour
		ellipse = cv2.fitEllipse(contour)
		(x, y), (MA, ma), angle = ellipse
		b = min(ma,MA)
		a = max(ma,MA)
		theta = np.deg2rad(angle)
		#computing ellipse perimeter using Ramanujan formula and area
		perimeter = np.pi * (1.5 * (a + b) - np.sqrt((1.5 * a + 0.5 * b) * (0.5 * a + 1.5 * b)))
		area = np.pi * a * b / 4
		
		if display_plot == True:
			fig, ax = plt.subplots(1)
			# Draw the ellipse
			output_img = np.dstack([binary_object] * 3)
			cv2.ellipse(output_img, (int(x), int(y)), (int(b/2), int(a/2)), angle, 0, 360, (255, 0, 0), 1)
			output_img = cv2.copyMakeBorder(output_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
			# add the binary mask with the ellipse to the plot
			ax.imshow(output_img)
			# invert the y-axis
			ax.invert_yaxis()
			plt.show()
		return a, b, theta , perimeter, area
	
	## B.3 Orientation measurments
	@staticmethod
	def long_axis_orientation(binary_image):
		"""
		Computes the orientation of a binary shape by fitting a rectangle around it and
		determining the angle between the longest side of the rectangle and the x-axis.
	
		Parameters:
		- binary_image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
	
		Returns:
		- angle: the angle of the shape's orientation, adjusted to ensure values fall within a practical range for interpretation.
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
			print("An error occured (long_axis_orientation): No contour found.")
			return None
	
	@staticmethod
	def compute_orientation_minimizing_second_moment(binary_image):
		"""
		Computes the orientation of a binary shape by minimizing the second moment of inertia.
		
		Parameters:
		- binary_image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- orientation_minimizing_second_moment: the angle of the shape's second moment axis, adjusted to ensure 
		  values fall within a practical range for interpretation.
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The ratio of the maximum Feret diameter to the minimum Feret diameter.
		"""
		max_feret, min_feret, _, _, _ = Basic_Parameters.feret_measurments(binary_object)
		return max_feret / min_feret
	
	@staticmethod
	def diameter_elongation(binary_object):
		"""
		Computes the elongation ratio between the minimum circumscribed circle and the maximum inscribed circle.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The ratio of the minimum circumscribed circle radius to the maximum inscribed circle radius.
		"""
		min_circ_radius, _ = Basic_Parameters.minimum_circumscribed_circle(binary_object)
		max_inscribed_radius, _ = Basic_Parameters.maximum_inscribed_circle(binary_object)
		return min_circ_radius / max_inscribed_radius
	
	@staticmethod
	def circularity(binary_object):
		"""
		Computes the circularity of a shape, measuring how close it is to a perfect circle.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
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
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The ratio of the shape's perimeter to the approximated ellipse perimeter.
		"""
		particle_perimeter = Basic_Parameters.particle_perimeter(binary_object)
		ellipse_perimeter = Basic_Parameters.equivalent_moment_ellipse(binary_object)[3]
		return particle_perimeter / ellipse_perimeter
	
	@staticmethod
	def ellipsoidity_area(binary_object):
		"""
		Computes the ellipsoidity of a shape based on its area.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The ratio of the shape's area to the equivalent moment ellipse area.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		ellipse_area = Basic_Parameters.equivalent_moment_ellipse(binary_object)[4]
		return particle_area / ellipse_area
	
	@staticmethod
	def polygon_to_circle_area(binary_object):
		"""
		Computes the ratio between the shape's area and the difference of the minimum circumscribed circle area and the maximum inscribed circle area. 
		This formula has been designed for this paper. Its role is to describe the difference in area between the minimum circumscribed circle
		and the maximum inscribed circle normalized to the particle area. This difference decreases with a higher number of sides for ideal shapes.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The computed ratio.
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		min_circ_radius, _ = Basic_Parameters.minimum_circumscribed_circle(binary_object)
		max_inscribed_radius, _ = Basic_Parameters.maximum_inscribed_circle(binary_object)
		return particle_area / ((np.pi * min_circ_radius ** 2) - (np.pi * max_inscribed_radius ** 2))
	
	@staticmethod
	def radius_form_index(binary_object, angle_step=5, display_plot=False):
		"""
		Calculates the radius form index (form index in the reference article) of a particle from its binary mask.
	
		Reference:
		Masad, E., Olcott, D., White, T., Tashman, L., 2001. Correlation of Fine Aggregate Imaging Shape Indices with Asphalt Mixture Performance. 
		Transportation Research Record 1757, 148–156. https://doi.org/10.3141/1757-17
	
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- angle_step (float): step size between two angles in degrees (default: 5).
		- display_plot (bool): if true a matplolib plot showing the radius interpolation is displayed.
	
		Returns:
		- form_index (float): radius form index of the particle.
		"""
		# Copy the binary image for processing
		thresh = binary_object.copy()
		# Find contours
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not contours:
			raise ValueError("An error occured (radius_form_index): No contour found.")
		# Assume the largest contour is the one of interest
		contour = max(contours, key=cv2.contourArea).squeeze()
		# Compute the centroid of the region
		M = cv2.moments(contour)
		if M["m00"] == 0:
			raise ValueError("An error occured (radius_form_index): contour has zero area, cannot compute centroid.")
		centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
		# Convert contour to polar coordinates
		rho, phi = Toolkit.cartesian_to_polar(contour,centroid)
		# Sort points by angle and ensure phi is in [0, 2π]
		sorted_indices = np.argsort(phi)
		rho = rho[sorted_indices]
		phi = np.mod(phi[sorted_indices], 2 * np.pi)
		# Interpolate radius values
		rho_interp = interp1d(phi, rho, kind='linear', fill_value='extrapolate', bounds_error=False)
		# Compute the Form Index
		angle_step_rad = np.radians(angle_step)  # Convert step size to radians
		angles = np.linspace(0, 2 * np.pi, int(360 / angle_step), endpoint=False)
		rho_values = rho_interp(angles)
		form_index = np.sum(np.abs(np.roll(rho_values, -1) - rho_values) / np.maximum(rho_values, 1e-6))
		# Plotting
		if display_plot == True:
			plt.figure()
			plt.plot(angles, rho_values, label='Interpolated Radius')
			plt.scatter(phi, rho, color='red', s=5, label='Original Radius')
			plt.xlabel('Angle (radians)')
			plt.ylabel('Radius')
			plt.title(f'Radius Interpolation for Form Index (Step: {angle_step}°)')
			plt.legend()
			plt.grid(True)
			plt.show()
		return form_index
		
	@staticmethod
	def independant_fourier_descriptors(binary_object, k=7, convexhull=False, order1=1, order2=12, norm=True):
		"""
		Computes the fourier amplitudes as independant descriptors. They can be normalized by a0 or not.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- k (int): determines the number of Fourier coefficients (N = 2^k).
		- convexhull (bool): whether to use the convex hull of the contour.
		- order1 (int): the first harmonic needed.
		- order2 (int): the last harmonic needed.
		- norm (bool): whether to normalize the amplitudes by a0.
		
		Returns:
		- Rtheta (list): list of the amplitudes between order1 and order2 for the particle using the Rtheta method.
		- Rtheta_ch (list): list of the amplitudes between order1 and order2 for the convex hull using the Rtheta method.
		- xy (list): list of the amplitudes between order1 and order2 for the particle using the XY method.
		- xy_ch (list): list of the amplitudes between order1 and order2 for the convex hull using the XY method.
		"""
		# Compute Fourier amplitudes using the Rtheta method (radius FFT)
		Rtheta_full = Fourier.radius_fft(binary_object, k, convexhull)
		if norm:
			Rtheta_full = Fourier.fourier_a0_normalization(Rtheta_full)
		Rtheta = Fourier.get_fft_order(Rtheta_full, order1, order2)
		# Compute Fourier amplitudes using the XY method (contour FFT)
		xy_full = Fourier.contour_fft(binary_object, k, convexhull)
		if norm:
			xy_full = Fourier.fourier_a0_normalization(xy_full)
		xy = Fourier.get_fft_order(xy_full, order1, order2)
		# If convex hull is used, compute the Fourier descriptors for the convex hull
		if convexhull:
			# Compute Fourier amplitudes using the Rtheta method (radius FFT)
			Rtheta_ch_full = Fourier.radius_fft(binary_object, k, convexhull)
			if norm:
				Rtheta_ch_full = Fourier.fourier_a0_normalization(Rtheta_ch_full)
			Rtheta_ch = Fourier.get_fft_order(Rtheta_ch_full, order1, order2)
			# Compute Fourier amplitudes using the XY method (contour FFT)
			xy_ch_full = Fourier.contour_fft(binary_object, k, convexhull)
			if norm:
				xy_ch_full = Fourier.fourier_a0_normalization(xy_ch_full)
			xy_ch = Fourier.get_fft_order(xy_ch_full, order1, order2)
		else:
			Rtheta_ch = None
			xy_ch = None
		return Rtheta, Rtheta_ch, xy, xy_ch
###############################################################################################################



################################### D. Roundness Descriptors Definitions ######################################
class Roundness_Descriptors:
	def __init__(self):
		pass

	@staticmethod
	def corner_focused_roundness(binary_object):
		"""
		Computes all the corner-focused roundness descriptors. To apply the method to every shapes, a modification has been made.
		The diameter measured at each angle is the diameter of the circle inscribed in the triangle formed by the corner, 
		instead of the diameter of curvature. They are inversely proportional.
		
		Reference:
		Wentworth, C.K., 1919. A Laboratory and Field Study of Cobble Abrasion. The Journal of Geology 27, 507–521.
		https://doi.org/10.1086/622676
		Wentworth, C.K., 1922. The shapes of beach pebbles. Professional Paper 75–83.
		https://doi.org/10.3133/pp131C
		Wadell, H., 1932. Volume, Shape, and Roundness of Rock Particles. The Journal of Geology 40, 443–451.
		https://doi.org/10.1086/623964
		Cailleux, A., 1942. Les actions éoliennes périglaciaires en Europe. Société géologique.
		Kuenen, Ph.H., 1956. Experimental Abrasion of Pebbles: 2. Rolling by Current. The Journal of Geology 64, 336–368.
		https://doi.org/10.1086/626370
		Lees, G., 1964. A New Method for Determining the Angularity of Particles. Sedimentology 3, 2–21.
		https://doi.org/10.1111/j.1365-3091.1964.tb00271.x)
		Dobkins, J.E., Folk, R.L., 1970. Shape development on Tahiti-Nui. Journal of Sedimentary Research 40, 1167–1203. 
		https://doi.org/10.1306/74D72162-2B21-11D7-8648000102C1865D
		Swan, B., 1974. Measures of particle roundness; a note. Journal of Sedimentary Research 44, 572–577. 
		https://doi.org/10.1306/74D72A90-2B21-11D7-8648000102C1865D
	
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- A dictionary with the following rounding values:
			- Wentworth (1919) roundness (float)
			- Wentworth (1922) roundness (float)
			- Wadell (1932) roundness (float)
			- Cailleux (1942) roundness (float)
			- Kuenen (1956) roundness (float)
			- Lees (1964) roundness (float)
			- Dobkins & Folk (1970) roundness (float)
			- Swan (1974) roundness (float)
		"""
		# Get all the needed information thanks to the extract_corner_metrics function
		particle_data = Toolkit.extract_corner_metrics(binary_object)
		# Extract values from the main dictionary
		r_ins = Basic_Parameters.minimum_circumscribed_circle(binary_object)[0]
		I = Basic_Parameters.feret_measurments(binary_object)[2] #Mean Feret diameter
		D_s = particle_data['diameter_sharpest_corner']
		L_s = particle_data['longest_distance']
		L = particle_data['max_feret']
		perpendicular_L = particle_data['perpendicular_feret']
		# Extract values from the corner list of dictionaries
		corner_list = particle_data['corners']
		N = len(corner_list)
		angles = [corner['angle_degrees'] for corner in corner_list]
		diameters = [corner['inscribed_circle_diameter'] for corner in corner_list]
		sorted_diameters = sorted(diameters, reverse=True)
		second_sharpest_diameter = sorted_diameters[1] if len(sorted_diameters) > 1 else D_s
		# Simple roundness descriptors
		wentworth_19 = D_s / L_s
		wentworth_22 = D_s / ((L+I)/2)
		cailleux = D_s / L
		kuenen = D_s / perpendicular_L
		dobkins_folk = D_s / (r_ins*2)
		swan = ((D_s + second_sharpest_diameter) / 2) / (r_ins*2)
		# Compute Lees and Wadell roundness
		N = len(corner_list)
		angles = [corner['angle_degrees'] for corner in corner_list]
		dist_center = [corner['center_to_corner'] for corner in corner_list]
		diameters = [corner['inscribed_circle_diameter'] for corner in corner_list]
		Lees_roundness = sum((180 - angle) * (dist / r_ins) for angle, dist in zip(angles, dist_center))
		Wadell_roundness = sum(diameter/(r_ins*2) for diameter in diameters)/N
		return {
			'Wentworth roundness 19': wentworth_19,
			'Wentworth roundness 22': wentworth_22,
			'Wadell roundness': Wadell_roundness,
			'Cailleux roundness': cailleux,
			'Kuenen roundness': kuenen,
			'Lees roundness': Lees_roundness,
			'Dobkins & Folk roundness': dobkins_folk,
			'Swan roundness': swan
		}
	
	@staticmethod
	def radius_angularity_index(image, display_plot=False):
		"""
		Computes the radius agularity index (Angularity index in the reference article).
		
		Reference:
		Masad, E., Olcott, D., White, T., Tashman, L., 2001. Correlation of Fine Aggregate Imaging Shape Indices with Asphalt Mixture Performance. 
		Transportation Research Record 1757, 148–156. https://doi.org/10.3141/1757-17
	
		Parameters:
		- image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- display_plot (bool): if true a matplolib plot showing the measurment is displayed.
	
		Returns:
		- index (float): the radius agularity index of the particle.
		"""
		region=measure.regionprops(image)[0]
		binary_object=image.copy()
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours,key=cv2.contourArea).squeeze()
		# convert contour to polar coordinates
		rho, phi = Toolkit.cartesian_to_polar(contour,(region.centroid[1],region.centroid[0]))
		# sort contour points by angle
		sorted_indices = np.argsort(phi)
		rho = rho[sorted_indices]
		phi = phi[sorted_indices]
		# ensure phi goes from 0 to 2pi
		phi = np.mod(phi, 2*np.pi)
		# interpolate radius values
		rho_interp = interp1d(phi, rho, kind='linear', fill_value='extrapolate')
		# calculate the semi-axes of the equivalent ellipse
		a = region.major_axis_length / 2.0
		b = region.minor_axis_length / 2.0
		theta = region.orientation
		
		index = 0
		for angle_rad in np.linspace(0, 2*np.pi, 360, endpoint=False):
			# get the radius at the current angle
			Rh = rho_interp(angle_rad)
			# calculate the distance from the center of the mask to the point on the equivalent ellipse
			REEh = a * b / np.sqrt((b * np.cos(angle_rad - theta))**2 + (a * np.sin(angle_rad - theta))**2)
			# update the index
			index += np.abs(Rh - REEh) / REEh
	
		if display_plot == True:
			fig, ax = plt.subplots(1)
			# draw the equivalent ellipse
			rr, cc = draw.ellipse_perimeter(int(region.centroid[0]), int(region.centroid[1]), int(a), int(b))
			# rotate the coordinates of the ellipse
			rr_rot = region.centroid[0] + (rr - region.centroid[0])*np.cos(region.orientation) - (cc - region.centroid[1])*np.sin(region.orientation)
			cc_rot = region.centroid[1] + (rr - region.centroid[0])*np.sin(region.orientation) + (cc - region.centroid[1])*np.cos(region.orientation)
			# ensure all drawn points are within image boundary
			rr_rot = np.clip(rr_rot, 0, binary_object.shape[0] - 1)
			cc_rot = np.clip(cc_rot, 0, binary_object.shape[1] - 1)
			# create an output image to draw on
			output_img = np.dstack([binary_object]*3) # create a 3-channel image
			# draw the ellipse on the image
			output_img[rr_rot.astype(int), cc_rot.astype(int), :] = [255, 0, 0]  # Use red for the ellipse
			# add the binary mask with the ellipse to the plot
			ax.imshow(output_img)
			# invert the y-axis
			ax.invert_yaxis()
			plt.title('Equivalent Ellipse Radius angularity Index')
			plt.show()
		return index
	
	@staticmethod
	def segment_angularity_index(image, precision=0.01, steps_bins_degree=10, num_points=50, method='rdp', distance_overlap=0.01, display_plot=False):
		"""
		Segment angularity index (Angularity using outline slope in other articles) of the binary object.
		
		Reference:
		Rao, C., Tutumluer, E., Kim, I.T., 2002. Quantification of Coarse Aggregate Angularity Based on Image Analysis. 
		Transportation Research Record 1787, 117–124. https://doi.org/10.3141/1787-13
	
		Parameters:
		- image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- precision (float): approximation accuracy parameter; for 'rdp', it’s the maximum distance from the contour to the approximated polygon
		  as a fraction of the contour perimeter; for 'rdp_uniform_cyril', it scales epsilon as precision times half the minor axis of the fitted ellipse.
		- steps_bins_degree (int): step size in degrees for the histogram bins of angle differences.
		- num_points (int or float): for 'linear', the number of points to sample uniformly along the contour;
		  for 'rdp_uniform_cyril', if <=1, it’s the ratio of points to sample, otherwise the number of points.
		- method (str): method to approximate the contour, one of 'rdp', 'linear', or 'rdp_uniform_cyril'.
		- distance_overlap (float): for 'rdp_uniform_cyril', the minimum distance threshold to filter points too close to approximated vertices.
		- display_plot (bool): if true a matplolib plot showing the contour and its approximation is displayed.
	
		Returns:
		- angularity_value (float): segment angularity index value of the binary object.
		"""
		binary_object = image.copy()
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		contour = max(contours, key=cv2.contourArea).squeeze()
	
		if method == 'rdp':
			if precision is not None:
				epsilon = precision * cv2.arcLength(contour, True)
				approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()
			else:
				approx = contour
				
		elif method == 'linear':
			# Calculate cumulative distance
			distances = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
			distances = np.insert(distances, 0, 0)
	
			# Determine interval and interpolate
			interval = distances[-1] / (num_points - 1)
			new_distances = np.arange(0, distances[-1], interval)
			approx = np.array([
				np.interp(new_distances, distances, contour[:, i]) for i in range(contour.shape[1])
			]).T
			
		elif method == 'rdp_uniform_cyril':
			ellipse = cv2.fitEllipse(contour)
			(x, y), (MA, ma), _ = ellipse
			a = min(ma,MA) / 2.0
			epsilon=precision*a
			approx = Toolkit.uniform_important_sampling(contour,epsilon,num_points,distance_overlap)
			
		else:
			raise ValueError("Invalid method (segment_angularity_index). Choose from 'rdp', 'linear', 'rdp_linear'.")
	
		if display_plot == True:
			plt.scatter(contour[:, 0], contour[:, 1], s=1)
			plt.scatter(approx[:, 0], approx[:, 1], s=1)
	
		vertex_angles = [Toolkit.angle(approx[i - 1], approx[i], approx[(i + 1) % len(approx)]) for i in range(len(approx))]
		vertex_angles_deg = [np.degrees(a) for a in vertex_angles]
		delta_angles_deg = [np.abs(vertex_angles_deg[i] - vertex_angles_deg[i - 1]) for i in range(len(vertex_angles))]
		histogram, _ = np.histogram(delta_angles_deg, bins=[steps_bins_degree * i for i in range(18)])
		angularity_value = sum([(i * steps_bins_degree) * histogram[i] for i in range(len(histogram))]) / len(approx)
		return angularity_value
	
	@staticmethod
	def gradient_angularity_index(image, display_plot=False):
		"""
		Calculates the gradient angularity index (gradient method in the reference article) for a binary object in an image.
		Averaging gives a result that's independent of object size, making it easier to compare grains.
		
		Reference:
		Tafesse, S., Robison Fernlund, J.M., Sun, W., Bergholm, F., 2013. Evaluation of image analysis methods used for quantification of particle angularity.
		Sedimentology 60, 1100–1110. https://doi.org/10.1111/j.1365-3091.2012.01367.x
		Chandan, C., Sivakumar, K., Masad, E., Fletcher, T., 2004. Application of Imaging Techniques to Geometry Analysis of Aggregate Particles. 
		Journal of Computing in Civil Engineering 18, 75–82. https://doi.org/10.1061/(ASCE)0887-3801(2004)18:1(75)
	
		Parameters:
		- image (numpy.ndarray): input image (grayscale or color) containing the object of interest.
		- display_plot (bool): if true a matplolib plot showing the contour and the binary image is displayed.
	
		Returns:
		- GI (float): the gradient angularity index value.
		"""
		binary_image = image.copy()
		# Detect the boundary (contour) of the object
		contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not contours:
			print("An error occured (gradient_angularity_index): no contours found in the image.")
			return None
		# Assume the largest contour is the object of interest
		contour = max(contours, key=cv2.contourArea).squeeze()
		if contour.ndim != 2:
			contour = contour.reshape(-1, 2)
		# Calculate the gradient vectors at each surface point using the Sobel operator
		binary_float = binary_image.astype(np.float32)
		Gx = cv2.Sobel(binary_float, cv2.CV_64F, 1, 0, ksize=3)
		Gy = cv2.Sobel(binary_float, cv2.CV_64F, 0, 1, ksize=3)
		# Calculate the angle of orientation at each point (in radians)
		theta = np.arctan2(Gy, Gx)
		# Extract the theta values at the contour points
		h_values = []
		for point in contour:
			x, y = point
			x = np.clip(x, 0, binary_image.shape[1] - 1)
			y = np.clip(y, 0, binary_image.shape[0] - 1)
			h = theta[y, x]
			h_values.append(h)
		h_values = np.array(h_values)
		# Unwrap the angle values to prevent discontinuities
		h_values_unwrapped = np.unwrap(h_values)
		# Calculate the differences Δh between points i and i+3
		delta_h = np.abs(h_values_unwrapped[:-3] - h_values_unwrapped[3:])
		GI = np.mean(delta_h)
		# Plot the contour over the binary image
		if display_plot == True:
		  plt.figure(figsize=(6, 6))
		  plt.imshow(binary_image, cmap='gray')
		  plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)
		  plt.title('Object Contour')
		  plt.axis('off')
		  plt.show()
		return GI
	
	@staticmethod
	def smoothing_angularity_index(binary_image, spline_degree=3, sub_spline_degree=3, intersection_downsample=5, first_bspline_samples=500,
								   second_bspline_samples=200, distance_eval_samples=200, max_mid_points=700, display_plot=False):
		"""
        Computes the smoothing angularity index (SAI) by analyzing contour smoothness using B-spline approximations.
    
        The process involves:
          1) Extracting the largest external contour as a polygon.
          2) Computing mid-points, downsampling if necessary, and fitting the first B-spline.
          3) Finding intersections with the polygon, downsampling, and fitting the second B-spline.
          4) Calculating true perpendicular distances from the second B-spline to the first.
          5) Computing SAI as the standard deviation of these distances.
        Note: Interpolation errors may arise from spline degree choices, sample sizes, or downsampling steps.
    
        Reference:
        Tafesse, S., Robison Fernlund, J.M., Sun, W., Bergholm, F., 2013. Evaluation of image analysis methods used for quantification of particle angularity.
        Sedimentology 60, 1100–1110. https://doi.org/10.1111/j.1365-3091.2012.01367.x
    
        Parameters:
        - binary_image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
        - spline_degree (int): degree of the first B-spline (default 3).
        - sub_spline_degree (int): degree of the second B-spline (default 3).
        - intersection_downsample (int): step size for downsampling intersection points before fitting the second B-spline (default 5).
        - first_bspline_samples (int): number of points to sample from the first B-spline (default 500).
        - second_bspline_samples (int): number of points to sample from the second B-spline and for distance calculations (default 200).
        - distance_eval_samples (int): unused parameter retained for compatibility; distances are computed over second_bspline_samples points.
        - max_mid_points (int): maximum number of mid-points to use; if exceeded, downsamples to this number (default 700).
        - display_plot (bool): if true, displays a matplotlib plot showing the interpolations and sample points (default False).
    
        Returns:
        - sai (float): the smoothing angularity index, computed as the standard deviation of the distances.
		"""
		# 1) Largest contour
		contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if len(contours) == 0:
			raise ValueError("An error occured (smoothing_angularity_index): no contour found in the image.")
		areas = [cv2.contourArea(c) for c in contours]
		max_idx = np.argmax(areas)
		contour = contours[max_idx].squeeze()
		if contour.ndim != 2 or contour.shape[1] != 2:
			raise ValueError(f"Contour shape unexpected. Found shape: {contour.shape}")
		# 2) Mid-points -> down-sample if too large -> first B-spline
		mid_pts = Toolkit.polygon_midpoints(binary_image,contour)
		# Down-sample if we have more than max_mid_points
		mid_pts_down = Toolkit.downsample_array(mid_pts, max_points=max_mid_points)
		# Increase smoothing to handle small fluctuations
		# The more points you have, the bigger s might need to be
		s_for_first_spline = 0.1 * len(mid_pts_down)  # Heuristic
		try:
			bspline1_pts, tck1 = Toolkit.bspline_fit(mid_pts_down, degree=spline_degree, num_samples=first_bspline_samples,
													   s=s_for_first_spline, closed=True)
			# 3) Intersections -> second B-spline
			intersection_pts = Toolkit.find_bspline_polygon_intersections(bspline1_pts, contour, closed=True)
			if len(intersection_pts) == 0:
				raise ValueError("An error occured (smoothing_angularity_index): no intersections found between the first B-spline and the polygon.")

			intersection_sub = Toolkit.downsample_points(intersection_pts, step=intersection_downsample)
			s_for_second_spline = 0.1 * len(intersection_sub)
			bspline2_pts, tck2 = Toolkit.bspline_fit(intersection_sub, degree=sub_spline_degree, num_samples=second_bspline_samples,
													   s=s_for_second_spline,closed=True)
		except Exception as e:
			# Log the error if needed
			print(f"An error occured (smoothing_angularity_index): {e}")
			return float("nan")
		# 4) Distances from second B-spline -> first B-spline
		distances = []
		for pt in tqdm(bspline2_pts, desc="Computing distances", disable=not display_plot):
			dist_pt = Toolkit.true_perp_distance_bspline(pt, tck1)
			distances.append(dist_pt)
		distances = np.array(distances)
		# 5) SAI = std(distances)
		sai = float(np.std(distances, ddof=1))
		# 6) Optional plot
		if display_plot == True:
			closed_contour = np.vstack([contour, contour[0]])
			plt.figure(figsize=(7, 7))
			plt.title("Smoothing angularity index visualization with Down-sampled Mid-Points")
			# Original contour
			plt.plot(closed_contour[:, 0], closed_contour[:, 1], 'k-', label='Original Contour')
			# First B-spline
			plt.plot(bspline1_pts[:, 0], bspline1_pts[:, 1], 'r--', label='First B-spline')
			# Second B-spline
			plt.plot(bspline2_pts[:, 0], bspline2_pts[:, 1], 'b--', label='Second B-spline')
			# Show intersections
			plt.scatter(intersection_pts[:, 0], intersection_pts[:, 1],
						c='orange', marker='x', label='Intersections')
			plt.scatter(intersection_sub[:, 0], intersection_sub[:, 1],
						c='green', marker='o', label='Downsampled Intersections')
			plt.axis('equal')
			plt.legend()
			plt.show()
		return sai
	
	@staticmethod
	def FRANG(image, k=10, contour_method=False):
		"""
        Computes the Rθ or XY Fourier roundness for harmonics 5 to 25 combined, termed FRANG (angularity in the original paper).
		The original article uses the Rθ method.
    
        Reference:
        Wang, L., Wang, X., Mohammad, L., Abadie, C., 2005. Unified Method to Quantify Aggregate Shape Angularity and Texture Using Fourier Analysis.
        Transportation Research Record: Journal of the Transportation Research Board, No. 1929, pp. 117-124.
    
        Parameters:
        - image (ndarray): binary image of the particle where the object is white (255) and background is black (0).
        - k (int): determines the number of Fourier coefficients (N = 2^k), should be at least 6 to cover 25 harmonics.
		- contour_method (bool): if false use the Rθ method (radius_fft) to obtain the Fourier spectrum,
						 if true use the XY method (contour_fft), default is false.
    
        Returns:
        - frang (float): the combined Fourier roundness value for harmonics 5 to 25.
		"""
		if contour_method == False:
			# Call radius_fft to compute Fourier coefficients
			_, fourier_coefficients = Fourier().radius_fft(image, k, return_full_spectrum=True)
		else:
			# Call contour_fft to compute Fourier coefficients
			_, fourier_coefficients = Fourier().contour_fft(image, k, return_full_spectrum=True)
		# Compute the descriptor
		a0 = np.real(fourier_coefficients[0])  # DC component
		frang = 0
		for i in range(5, 26):  # Angularity range
			a, b = np.real(fourier_coefficients[i]), np.imag(fourier_coefficients[i])
			frang += (a / a0)**2 + (b / a0)**2
		return frang

	@staticmethod
	def angularity_factor(image_2d, order=5, mode="a0_normalisation"):
		"""
		Computes the Angularity Factor of a 2D input using the method of Sun et al. (2012),
		but without using fftshift. We take the first-quadrant frequencies in an unshifted FFT.

		Reference:
		Sun, W., Wang, L., Tutumluer, E., 2012. Image Analysis Technique for Aggregate Morphology Analysis with Two-Dimensional Fourier Transform Method.
		Transportation Research Record 2267, 3–13. https://doi.org/10.3141/2267-01

		Parameters: 
		- image_2d (2D ndarray, shape: N×M): the input array (particle surface, etc.).
		- order (int): the maximum 'frequency index' in each dimension away from DC (excluding DC), i.e. p, q in [1..order].
		- mode {"no_normalisation", "mean_normalisation", "a0_normalisation"}: how to normalize the FFT coefficients before summing.
	
		Returns:
		- angularity (float): the resulting angularity factor.
		"""
		# Perform the unshifted 2D FFT
		F = fft2(image_2d)
		# Identify the DC component at (0,0)
		a0 = np.real(F[0, 0])  # Real part of DC
		b0 = np.imag(F[0, 0])  # Imaginary part of DC (usually near zero for a real image)
		# Get the dimensions of the image
		N, M = F.shape
		# Determine how far we can go in each dimension. We don't want to exceed half the size in either dimension.
		# E.g., for the row dimension, valid "positive freq" indices might be 1..(N//2).
		row_limit = min(order, N // 2)
		col_limit = min(order, M // 2)
		# Extract the first-quadrant frequencies, skipping (0,0). So rows [1..row_limit], columns [1..col_limit].
		quadrant_coeffs = F[1 : row_limit + 1, 1 : col_limit + 1]
		# Flatten to iterate easily
		quadrant_coeffs_flat = quadrant_coeffs.flatten()
		# Accumulate the sum with chosen normalization
		angularity = 0.0
	
		if mode == "no_normalisation":
			for amp in quadrant_coeffs_flat:
				a, b = np.real(amp), np.imag(amp)
				angularity += a**2 + b**2
	
		elif mode == "mean_normalisation":
			# Compute the mean among those quadrant coefficients
			mean_val = np.mean(quadrant_coeffs_flat)
			amean, bmean = np.real(mean_val), np.imag(mean_val)
	
			for amp in quadrant_coeffs_flat:
				a, b = np.real(amp), np.imag(amp)
				# Guard against zero mean
				if amean != 0 and bmean != 0:
					angularity += (a / amean)**2 + (b / bmean)**2
				else:
					# If the mean is 0, fallback to no_normalisation
					angularity += a**2 + b**2
	
		elif mode == "a0_normalisation":
			# Normalize by the real DC component a0
			for amp in quadrant_coeffs_flat:
				a, b = np.real(amp), np.imag(amp)
				if a0 != 0:
					angularity += (a / a0)**2 + (b / a0)**2
				else:
					# Fallback if DC is zero
					angularity += a**2 + b**2
		return angularity
##############################################################################################################



################################### E. Roughness Descriptors Definitions #####################################
class Roughness_Descriptors:
	def __init__(self):
		pass
	
	@staticmethod
	def FRTXTR(image, k=10, contour_method=False):
		"""
		Computes the Rθ or XY Fourier rounghness 26-180 combined (texture in the original paper) of the particle.
		The original article uses the Rθ method.
		
		Reference:
		Wang, L., Wang, X., Mohammad, L., Abadie, C., 2005. Unified Method to Quantify Aggregate Shape Angularity and Texture Using Fourier Analysis. 
		Journal of Materials in Civil Engineering 17, 498–504. https://doi.org/10.1061/(ASCE)0899-1561(2005)17:5(498)
	
		Parameters:
		- image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- k (int): determines the number of Fourier coefficients (N = 2^k).
		- contour_method (bool): if false use the Rθ method (radius_fft) to obtain the Fourier spectrum,
								 if true use the XY method (contour_fft), default is false.
	
		Returns:
		- frxtr (float): the combined Fourier roundness value for harmonics 26 to 180.
		"""
		if contour_method == False:
			# Call radius_fft to compute Fourier coefficients
			_, fourier_coefficients = Fourier().radius_fft(image, k, return_full_spectrum=True)
		else:
			# Call contour_fft to compute Fourier coefficients
			_, fourier_coefficients = Fourier().contour_fft(image, k, return_full_spectrum=True)
		# Compute the descriptor
		a0 = np.real(fourier_coefficients[0])  # DC component
		frxtr = 0
		for i in range(26, 181):  # Texture range
			a, b = np.real(fourier_coefficients[i]), np.imag(fourier_coefficients[i])
			frxtr += (a / a0)**2 + (b / a0)**2
		return frxtr
	
	@staticmethod
	def erosion_dilation_ratio(image, kernel_size=3,iter_erode=20,iter_dilate=20, display_plot=False):
		"""
		Computes the erosion–dilation ratio (surface texture index in the reference article) of the binary object.
		
		Reference:
		Al-Rousan, T., Masad, E., Tutumluer, E., Pan, T., 2007. Evaluation of image analysis techniques for quantifying aggregate shape characteristics. 
		Construction and Building Materials 21, 978–990. https://doi.org/10.1016/j.conbuildmat.2006.03.005
		
		Parameters:
		- image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- kernel_size (int): size of the kernel for erosion and dilation operation.
		- iter_erode (int): number of erosion iteration.
		- iter_dilate (int): number of dilation iteration.
		- display_plot (bool): if true a matplolib plot showing the eroded and dilated grain images are displayed.
	
		Returns:
		- surface_parameter (float): erosion–dilation ratio of the binary object.
		"""
		binary_image = image.copy()
		# Ensure the image is binary (only 0 and 255 values)
		assert set(np.unique(binary_image)).issubset({0, 255}), "An error occured (erosion_dilation_ratio): image should be binary with 0 and 255 values only."
		# Define the kernel for the morphological operations
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		# Erode the image
		eroded = cv2.erode(binary_image, kernel, iterations=iter_erode)
		# Dilate the eroded image
		dilated = cv2.dilate(eroded, kernel, iterations=iter_dilate)
		# Show the figure with the eroded and dilated images
		if display_plot == True:
			fig,axs=plt.subplots(1,3)
			axs[0].imshow(binary_image)
			axs[1].imshow(eroded)
			axs[2].imshow(dilated)
		# Compute areas A1 and A2
		A1 = np.sum(binary_image == 255)
		A2 = np.sum(dilated == 255)
		# Calculate the surface parameter
		surface_parameter = (A1 - A2) / A1 * 100
		return surface_parameter
	
	@staticmethod
	def morphological_fractal(image, max_cycles=10, display_plot=False):
		"""
        Computes the morphological fractal dimension of a particle using the Fractal-Behavior Technique.
        This method analyzes the particle's boundary complexity by applying multiple cycles of erosion and dilation
        to a binary image, then fitting a linear regression to the log-log plot of cycles versus effective widths.
        The slope of this fit represents the fractal dimension, a measure of boundary roughness or complexity.

        Reference:
        Masad, E., Button, J.W., Papagiannakis, T., 2000. Fine-Aggregate Angularity: Automated Image Analysis Approach.
        Transportation Research Record 1721, 66–72. https://doi.org/10.3141/1721-08

        Parameters:
        - image (ndarray): Binary image of the particle, where the particle is white (255) and the background is black (0).
        - max_cycles (int): Maximum number of erosion-dilation cycles to perform. Determines the range over which the
          fractal behavior is analyzed. Default is 10.
        - display_plot (bool): If True, displays a log-log plot of cycles vs. effective widths. Default is False.

        Returns:
        - slope (float): The morphological fractal dimension of the particle, calculated as the slope of the linear
          regression on the log-log plot of cycles versus effective widths.
        """
		binary_image = image.copy()
		kernel = np.ones((3, 3), np.uint8)
		widths = []
		cycles = list(range(1, max_cycles + 1))
	
		for cycle in cycles:
			eroded = cv2.erode(binary_image, kernel, iterations=cycle)
			dilated = cv2.dilate(binary_image, kernel, iterations=cycle)
			# Ex-OR operation
			xor_image = cv2.bitwise_xor(eroded, dilated)
			# Find contours in the xor_image
			contours, _ = cv2.findContours(xor_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
			# Calculate the total length of all contours (both external and internal)
			boundary_length = sum(cv2.arcLength(contour, True) for contour in contours)
			# Measure effective width: (total number of changed pixels) / (boundary length x cycle)
			total_pixels = np.sum(xor_image == 255)
			width = total_pixels / (boundary_length * cycle)
			widths.append(width)
	
		if display_plot == True:
			# Plotting on a log-log scale
			plt.loglog(cycles, widths, marker='o', linestyle='-')
			plt.xlabel('Number of Erosion-Dilation Cycles')
			plt.ylabel('Effective Width')
			plt.show()
	
		# Fractal dimension estimation
		log_cycles = np.log(cycles)
		log_widths = np.log(widths)
		slope, _ = np.polyfit(log_cycles, log_widths, 1)
		return slope
	
	@staticmethod
	def area_convexity(binary_object):
		"""
		Computes the area convexity of the object, the formula is designed to divide the large number by the small number.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The area convexity of the object (float).
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		ch_area = Basic_Parameters.convexhull_area_perimeter(binary_object)[0]
		return ch_area/particle_area
	
	@staticmethod
	def area_convexity_percentage(binary_object):
		"""
		Computes the area convexity percentage of the object, the formula is designed to subtract the large number by the small number and avoid negative values.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The area convexity percentage of the object (float).	
		"""
		particle_area = Basic_Parameters.particle_area(binary_object)
		ch_area = Basic_Parameters.convexhull_area_perimeter(binary_object)[0]
		return (ch_area - particle_area)/particle_area
	
	@staticmethod
	def perimeter_convexity(binary_object):
		"""
		Computes the perimeter convexity of the object, the formula is designed to divide the large number by the small number.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The perimeter convexity of the object (float).
		"""
		particle_perimeter = Basic_Parameters.particle_perimeter(binary_object)
		ch_perimeter = Basic_Parameters.convexhull_area_perimeter(binary_object)[1]
		return particle_perimeter/ch_perimeter
	
	@staticmethod
	def perimeter_convexity_percentage(binary_object):
		"""
		Computes the perimeter convexity percentage of the object, the formula is designed to subtract the large number by the small number
		and avoid negative values.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		
		Returns:
		- The perimeter convexity percentage of the object (float).
		"""
		particle_perimeter = Basic_Parameters.particle_perimeter(binary_object)
		ch_perimeter = Basic_Parameters.convexhull_area_perimeter(binary_object)[1]
		return (particle_perimeter-ch_perimeter)/particle_perimeter
	
	@staticmethod
	def vertex_concavity(image, threshold=0.01, display_plot=False):
		"""
		Calculates the vertex concavity (Ω-value in the original article) for a binary object in an image by analyzing the angles at each vertex of its outline.
		
		Reference:
		Heilbronner, R., Keulen, N., 2006. Grain size and grain shape analysis of fault rocks. Tectonophysics, Deformation mechanisms, 
		microstructure and rheology of rocks in nature and experiment 427, 199–216. https://doi.org/10.1016/j.tecto.2006.05.020
	
		Parameters:
		- image  (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- threshold (float): threshold value for selecting contour interest point based correponding to the percentage of contour length.
		- display_plot (bool): if true a matplolib plot showing the contour over the binary image with its vertices is displayed.
		
		Returns:
		- omega_value (float): the vertex concavity, representing the percentage of concave angles in the particle's outline.
		"""
		# Step 1: Detect the boundary (contour) of the object
		contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not contours:
			print("An error occured (vertex_concavity): no contours found in the image.")
			return None
		# Assume the largest contour is the object of interest
		contour = max(contours, key=cv2.contourArea).squeeze()
		# Simplify the contour to identify significant vertices
		epsilon = threshold * cv2.arcLength(contour, True)
		contour = cv2.approxPolyDP(contour, epsilon, True)
		contour = contour.squeeze()
		
		if contour.ndim != 2:
			contour = contour.reshape(-1, 2)
		# Step 2: Identify vertices (contour points)
		# For this method, we'll consider all contour points as vertices
		num_vertices = len(contour)
		# Step 3: Calculate angles at each vertex
		concave_angles = 0
		for i in range(num_vertices):
			# Get three consecutive points: A, B, C
			A = contour[i - 1]
			B = contour[i]
			C = contour[(i + 1) % num_vertices]
			# Vectors BA and BC
			BA = A - B
			BC = C - B
			# Calculate the angle between BA and BC
			cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
			angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Angle in radians
			# Determine if the angle is concave or convex using the cross product
			cross = BA[0]*BC[1] - BA[1]*BC[0]
			if cross < 0:
				# Concave angle
				concave_angles += 1
		# Step 4: Calculate Ω-value
		omega_value = (concave_angles / num_vertices) * 100
		# Plot the contour over the binary image with Vertices
		if display_plot == True:
			plt.figure(figsize=(6, 6))
			plt.imshow(image, cmap='gray')
			plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)
			plt.title('Particle Outline with Vertices')
			plt.axis('off')
			plt.show()
		return omega_value
	
	@staticmethod
	def area_perimeter_fractal(binary_image, min_scale=0.1, max_scale=1.0, num_scales=10, display_plot=False):
		"""
		Calculates the fractal dimension of an object in a binary image using the area-perimeter method.
		
		Reference:
		Hyslip, J.P., Vallejo, L.E., 1997. Fractal analysis of the roughness and size distribution of granular materials. 
		Engineering Geology, Fractals in Engineering Geology 48, 231–244. https://doi.org/10.1016/S0013-7952(97)00046-X
	
		Parameters:
		- binary_image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- min_scale (float): minimum scale factor for resizing. Defaults to 0.1.
		- max_scale (float): maximum scale factor for resizing (original size). Defaults to 1.0.
		- num_scales (int): number of scaling factors to use between min_scale and max_scale. Defaults to 10.
		- display_plot (bool): whether to display the log-log plot.
	
		Returns:
		- fractal_dimension (float): estimated fractal dimension of the object.
		"""
		# Ensure binary image is single channel
		if len(binary_image.shape) > 2:
			binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
		# Validate binary image
		unique_values = np.unique(binary_image)
		if not set(unique_values).issubset({0, 255}):
			raise ValueError("An error occured (area_perimeter_fractal): input image must be binary with values 0 and 255.")
		# Optionally invert the image if background is white
		if np.mean(binary_image) > 127:
			binary_image = cv2.bitwise_not(binary_image)
		# Initialize lists to store measurements
		areas = []
		perimeters = []
		scales = []
		# Generate scaling factors
		scales_factors = np.linspace(max_scale, min_scale, num=num_scales)
	
		for scale in scales_factors:
			# Resize image using nearest neighbor to preserve binary nature
			resized_image = cv2.resize(binary_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
			# Find contours
			contours, _ = cv2.findContours(resized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			
			if contours:
				# Aggregate area and perimeter for all contours
				total_area = sum(cv2.contourArea(cnt) for cnt in contours)
				total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
				# Avoid zero area or perimeter
				if total_area > 0 and total_perimeter > 0:
					areas.append(total_area)
					perimeters.append(total_perimeter)
					scales.append(scale)
		# Check if sufficient data points were collected
		if len(areas) < 2:
			raise ValueError("An error occured (area_perimeter_fractal): insufficient data points for regression. Try adjusting the scaling parameters.")
		# Convert to logarithmic scale
		log_areas = np.log(areas)
		log_perimeters = np.log(perimeters)
		# Perform linear regression
		slope, intercept = np.polyfit(log_areas, log_perimeters, 1)
		# Calculate fractal dimension
		fractal_dimension = 2 / slope
	
		if display_plot:
			# Plot the data points
			plt.figure(figsize=(8, 6))
			plt.plot(log_areas, log_perimeters, 'bo', label='Data Points')
			# Plot the fitted regression line
			fitted_line = slope * log_areas + intercept
			plt.plot(log_areas, fitted_line, 'r--', label=f'Fit: slope={slope:.2f}')
			# Annotations and labels
			plt.xlabel('log(Area)')
			plt.ylabel('log(Perimeter)')
			plt.title(f'Fractal Dimension Calculation\nSlope: {slope:.2f}, Fractal Dimension: {fractal_dimension:.2f}')
			plt.legend()
			plt.grid(True)
			plt.show()
		return fractal_dimension
	
	@staticmethod
	def box_counting(image, min_box_size=2, max_box_size=64):
		"""
		Calculates the fractal dimension of an object in a binary image using the box counting method.
		
		Reference:
		Asvestas, P., Matsopoulos, G.K., Nikita, K.S., 1999. Estimation of fractal dimension of images using a fixed mass approach. 
		Pattern Recognition Letters 20, 347–354. https://doi.org/10.1016/S0167-8655(99)00004-5
	
		Parameters:
		- binary_image (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- min_box_size (int): minimum box size (e.g., 2).
		- max_box_size (int): maximum box size (e.g., 64).
	
		Returns:
		- fractal_dimension (float): fractal dimension value.
		"""
		# Get dimensions of the image
		height, width = image.shape
		box_counts = []
		box_sizes = []
		# Iterate over different box sizes
		for box_size in range(min_box_size, max_box_size + 1, 2):
			# Count the number of boxes that contain part of the fractal
			box_count = 0
			for i in range(0, height, box_size):
				for j in range(0, width, box_size):
					# Check if there are any pixels belonging to the object in the current box
					if np.any(image[i:i + box_size, j:j + box_size]):
						box_count += 1
			# Record results
			box_counts.append(box_count)
			box_sizes.append(box_size)
		# Convert to log scale
		log_box_sizes = np.log(box_sizes)
		log_box_counts = np.log(box_counts)
		# The fractal dimension corresponds to the slope of the line in the log-log plot
		coeffs = np.polyfit(log_box_sizes, log_box_counts, 1)
		fractal_dimension = -coeffs[0]
		return fractal_dimension
	
	@staticmethod
	def wavelet_texture(image, wavelet_name='bior4.4'):
		"""
		Computes the energy from the six-level wavelet decomposition. 
		The bior4.4 wavelet function is the closest to the CDF 9/7 function used in the reference article 
		and has the advantage of being available in the PyWavelets library.
		
		Reference:
		Chandan, C., Sivakumar, K., Masad, E., Fletcher, T., 2004. Application of Imaging Techniques to Geometry Analysis of Aggregate Particles.
		Journal of Computing in Civil Engineering 18, 75–82. https://doi.org/10.1061/(ASCE)0887-3801(2004)18:1(75)
	
		Parameters:
		- image (ndarray): 2D numpy array representing the image.
		- wavelet_name (str): the name of the wavelet function to use.
	
		Returns:
		- energy_level_6 (float): energy from the six-level wavelet decomposition.
		"""
		num_levels = 6
		features_per_level = Roughness_Descriptors.compute_wavelet_features(image, num_levels, wavelet_name)
		# Extract energy value from level 6
		energy_level_6 = features_per_level[-1]['High_features_combined']['energy']
		return energy_level_6
	
	@staticmethod
	def compute_wavelet_features(image, num_levels=3, wavelet_name='bior4.4'):
		"""
		Computes statistical features from wavelet decomposition up to the specified number of levels.

		Parameters:
		- image (ndarray): 2D numpy array representing the image.
		- num_levels (int): the number of decomposition levels; must be a positive integer up to the maximum allowed by the image size.
		- wavelet_name (str): the name of the wavelet function to use.

		Returns:
		- features_per_level (list): list of dictionaries containing features for each decomposition level.
			Each dictionary includes:
			- 'level' (int): the decomposition level.
			- 'Low-Low_features' (dict): statistics of the approximation coefficients.
			- 'High_features_combined' (dict): averaged statistics of the detail coefficients.
			The statistics includes:
			- the energy.
			- the entropy.
			- the contrast.
			- the mean and the standard deviation.
		"""
		# Obtain wavelet feature maps
		feature_maps = Toolkit.wavelet_feature_maps(image, num_levels, wavelet_name)
		features_per_level = []
	
		for level, coeffs in enumerate(feature_maps, start=1):
			cA, cH, cV, cD = coeffs
			# Compute features for Low-Low (approximation coefficients)
			cA_features = Toolkit.compute_statistics(cA)
			# Compute statistics for each detail subband
			cH_features = Toolkit.compute_statistics(cH)
			cV_features = Toolkit.compute_statistics(cV)
			cD_features = Toolkit.compute_statistics(cD)
			# Average the statistics across detail subbands, the mean of mean is possible, the image have strictly 
			detail_features = {
				'energy': np.mean([cH_features['energy'], cV_features['energy'], cD_features['energy']]),
				'entropy': np.mean([cH_features['entropy'], cV_features['entropy'], cD_features['entropy']]),
				'mean': np.mean([cH_features['mean'], cV_features['mean'], cD_features['mean']]),
				'std_dev': np.mean([cH_features['std_dev'], cV_features['std_dev'], cD_features['std_dev']]),
				'contrast': np.mean([cH_features['contrast'], cV_features['contrast'], cD_features['contrast']])
			}
			# Store features for this level
			features_per_level.append({
				'level': level,
				'Low-Low_features': cA_features,
				'High_features_combined': detail_features
			})
		return features_per_level
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
		- ndarray: Fourier amplitudes (and optionally the full spectrum).
		"""
		# Compute region properties
		regions = measure.regionprops(image)
		if not regions:
			raise ValueError("An error occured (radius_fft): no regions found in the image.")
		region = regions[0]
		# Find contours
		binary_object = image.copy()
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not contours:
			raise ValueError("An error occured (radius_fft): no contours found in the image.")
		# Select the largest contour based on area
		largest_contour = max(contours, key=cv2.contourArea)
		# Reshape contour to (n_points, 2)
		contour = largest_contour.squeeze()
		if convexhull:
			# Compute convex hull
			contour_hull = cv2.convexHull(largest_contour).squeeze()
			if contour_hull.ndim != 2 or contour_hull.shape[1] != 2:
				raise ValueError("An error occured (radius_fft): convex hull has an unexpected shape after squeezing.")
			# Calculate centroid using image moments
			moments = cv2.moments(contour_hull)
			if moments["m00"] == 0:
				raise ValueError("An error occured (radius_fft): zero division error while calculating centroid from moments.")
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
		# Optionally return the entire Fourier Spectrum
		if return_full_spectrum:
			spectrum = fourier_transform[:N//2]
			return fourier_amplitudes[:N//2], spectrum
		else:
			return fourier_amplitudes[:N//2]
	
	@staticmethod
	def contour_fft(image, k=7, convexhull=False, return_full_spectrum=False, display_plot=False):
		"""
		Computes the Fourier Transform of the contour of an object in a binary image using x and y positions (XY method).
		
		Parameters:
		- image (ndarray): binary image of the particle.
		- k (int): determines the number of Fourier coefficients (N = 2^k).
		- convexhull (bool): whether to use the convex hull of the contour.
		- return_full_spectrum (bool): whether to return the full Fourier spectrum.
		- display_plot (bool): if true a matplolib plot showing the convex hull is displayed.
		
		Returns:
		- ndarray: Fourier amplitudes (and optionally the full spectrum).
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
			raise ValueError("An error occured (contour_fft): contour array has an unexpected shape.")
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
		fourier_amplitudes = np.abs(fourier_transform) / N
		# Optionally return the entire Fourier Spectrum
		if return_full_spectrum:
			spectrum = fourier_transform[:N//2]
			return fourier_amplitudes[:N//2], spectrum
		else:
			return fourier_amplitudes[:N//2]

	@staticmethod
	def fourier_a0_normalization(amplitudes):
		"""
		Normalizes a list or array of Fourier amplitudes by dividing each value by the first amplitude (a0).
		
		Parameters:
		- amplitudes (list or numpy array): a list or array of amplitude values where the first element is a0.
		
		Returns:
		- numpy array: a new array where each amplitude is divided by a0, the new a0 is 1.
		"""
		amplitudes = np.array(amplitudes)
		a0 = amplitudes[0]
		if np.any(a0 == 0):  # Check if a0 contains zero
			raise ValueError("An error occured (fourier_a0_normalization): zero division error, the first amplitude (a0) must be non-zero for normalization.")
		return amplitudes / a0  # Element-wise division using NumPy
	
	@staticmethod
	def get_fft_order(amplitudes,order1,order2):
		"""
		Return the Fourier amplitudes within a certain range (order1 to order2).

		Parameters:
		- amplitudes (list): a list of Fourier amplitude values.

		Returns:
		- The values in a list between the two orders.
		"""
		return amplitudes[order1 : order2+1]
###############################################################################################################



###################################  G. Toolkit Definitions ###################################################
class Toolkit:
	def __init__(self):
		pass

	@staticmethod
	def extract_corner_metrics(image, epsilon=0.01, display_plot=False):
		"""
		The extract_corner_metrics function analyzes binary images to extract key corner properties 
		to compute corner-focused roundness descriptors. It provides precise geometric 
		measurements by identifying significant corners and reducing roughness effects.
		
		Parameters:
		- image (ndarray): binary image where the object is white (255) and background is black (0).
		- epsilon (float): approximation accuracy as a fraction of the contour perimeter for the RDP algorithm.
		- display_plot (bool): if true, displays a matplotlib plot showing measurements for corner-focused roundness descriptors.
		
		Returns:
		- output (dict): dictionary containing:
			- max_feret (float): maximum Feret diameter.
			- max_feret_angle (float): angle of the maximum Feret diameter in degrees.
			- perpendicular_feret (float): Feret diameter perpendicular to the maximum Feret.
			- perpendicular_feret_angle (float): angle of the perpendicular Feret diameter in degrees.
			- diameter_sharpest_corner (float): diameter of the inscribed circle at the sharpest corner.
			- longest_distance (float): longest distance from the sharpest corner to any contour point.
			- corners (list): list of dictionaries with properties for each corner (corner_point, angle_degrees, etc.).
		"""
		# Ensure the image is in grayscale
		if len(image.shape) != 2:
			raise ValueError("An error occured (extract_corner_metrics): input image must be a binary (grayscale) image.")
	
		binary_object = image.copy()
		# Find contours
		contours, _ = cv2.findContours(binary_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if not contours:
			raise ValueError("An error occured (extract_corner_metrics): no contours found in the image.")
		# Assume the largest contour is the one of interest
		contour = max(contours, key=cv2.contourArea).squeeze()
		# Ensure contour is in the correct shape
		if len(contour.shape) != 2:
			contour = contour.reshape(-1, 2)
		# Approximate the contour to get the summits (corners) using RDP algorithm
		perimeter = cv2.arcLength(contour, True)
		approx_epsilon = epsilon * perimeter
		approx = cv2.approxPolyDP(contour, approx_epsilon, True).squeeze()
		ins_y, ins_x = Basic_Parameters.maximum_inscribed_circle(image)[1]
	
		if len(approx.shape) != 2:
			approx = approx.reshape(-1, 2)
	
		summits = approx
		# Find the indices of the summits in the contour
		summit_indices = []
		for summit_point in summits:
			distances = np.linalg.norm(contour - summit_point, axis=1)
			index_in_contour = np.argmin(distances)
			summit_indices.append(index_in_contour)
		# Iterate through summits to find the sharpest internal angle
		min_angle = np.inf
		sharpest_vertex = None
		sharpest_index = -1
		sharpest_points = None
		for i in range(len(summit_indices)):
			s_idx = summit_indices[i]
			prev_s_idx = summit_indices[(i - 1) % len(summit_indices)]
			next_s_idx = summit_indices[(i + 1) % len(summit_indices)]
			# Calculate the minimum distance between the summit and the previous and next summits
			distance_prev = Toolkit.distance_along_contour_reverse(contour, s_idx, prev_s_idx)
			distance_next = Toolkit.distance_along_contour(contour, s_idx, next_s_idx)
			dmin = min(distance_prev, distance_next)
			# Find the points at dmin distance from the summit
			point_prev = Toolkit.find_point_at_distance(contour, s_idx, dmin, direction='backward')
			point_next = Toolkit.find_point_at_distance(contour, s_idx, dmin, direction='forward')
	
			summit_point = contour[s_idx]
			theta = Toolkit.compute_internal_angle(point_prev, summit_point, point_next)
	
			if theta < min_angle:
				min_angle = theta
				sharpest_vertex = summit_point
				sharpest_index = s_idx
				sharpest_points = (point_prev, summit_point, point_next)
		# Compute the inscribed circle at the sharpest corner
		A, V, B = sharpest_points
		radius, center = Toolkit.compute_inscribed_circle_triangle(A, V, B)
		diameter_sharpest_corner = 2 * radius
		# Find the farthest point on the contour from the sharpest corner
		distances = np.linalg.norm(contour - sharpest_vertex, axis=1)
		max_distance_idx = np.argmax(distances)
		farthest_point = contour[max_distance_idx]
		max_distance = distances[max_distance_idx]
		# Compute Maximum and Perpendicular Feret Diameters
		feret_diameters = Toolkit.compute_feret_diameters(contour)
		# Identify Maximum Feret
		max_feret_angle = max(feret_diameters, key=feret_diameters.get)
		max_feret = feret_diameters[max_feret_angle]
		# Compute Perpendicular Feret
		perpendicular_angle = (max_feret_angle + 90) % 180
		# To ensure the perpendicular angle is within 0-179 degrees
		perpendicular_angle = perpendicular_angle if perpendicular_angle < 180 else perpendicular_angle - 180
		perpendicular_feret = feret_diameters.get(perpendicular_angle, 0)
		# Compute endpoints for Maximum Feret and center the contour
		center_contour = np.mean(contour, axis=0)
		centered_contour = contour - center_contour
		# Rotation matrix for Maximum Feret
		theta_max = np.deg2rad(max_feret_angle)
		rotation_matrix_max = np.array([
			[np.cos(theta_max), np.sin(theta_max)],
			[-np.sin(theta_max), np.cos(theta_max)]
		])
		# Rotate centered contour
		rotated_contour_max = centered_contour @ rotation_matrix_max
		# Find min and max x in rotated contour
		min_x_max = np.min(rotated_contour_max[:, 0])
		max_x_max = np.max(rotated_contour_max[:, 0])
		# Find the indices of the points with min and max x
		min_x_idx_max = np.argmin(rotated_contour_max[:, 0])
		max_x_idx_max = np.argmax(rotated_contour_max[:, 0])
		# Get the actual points
		min_point_rotated_max = rotated_contour_max[min_x_idx_max]
		max_point_rotated_max = rotated_contour_max[max_x_idx_max]
		# Rotate back to original coordinates
		point1_max = min_point_rotated_max @ rotation_matrix_max.T + center_contour
		point2_max = max_point_rotated_max @ rotation_matrix_max.T + center_contour
		# Compute endpoints for Perpendicular Feret
		theta_perp = np.deg2rad(perpendicular_angle)
		rotation_matrix_perp = np.array([
			[np.cos(theta_perp), np.sin(theta_perp)],
			[-np.sin(theta_perp), np.cos(theta_perp)]
		])
		# Rotate centered contour
		rotated_contour_perp = centered_contour @ rotation_matrix_perp
		# Find min and max x in rotated contour
		min_x_perp = np.min(rotated_contour_perp[:, 0])
		max_x_perp = np.max(rotated_contour_perp[:, 0])
		# Find the indices of the points with min and max x
		min_x_idx_perp = np.argmin(rotated_contour_perp[:, 0])
		max_x_idx_perp = np.argmax(rotated_contour_perp[:, 0])
		# Get the actual points
		min_point_rotated_perp = rotated_contour_perp[min_x_idx_perp]
		max_point_rotated_perp = rotated_contour_perp[max_x_idx_perp]
		# Rotate back to original coordinates
		point1_perp = min_point_rotated_perp @ rotation_matrix_perp.T + center_contour
		point2_perp = max_point_rotated_perp @ rotation_matrix_perp.T + center_contour
		# Compute properties for all corners
		corner_properties = []
	
		for i in range(len(summit_indices)):
			s_idx = summit_indices[i]
			prev_s_idx = summit_indices[(i - 1) % len(summit_indices)]
			next_s_idx = summit_indices[(i + 1) % len(summit_indices)]
			# Calculate the minimum distance between the summit and the previous and next summits
			distance_prev = Toolkit.distance_along_contour_reverse(contour, s_idx, prev_s_idx)
			distance_next = Toolkit.distance_along_contour(contour, s_idx, next_s_idx)
			dmin = min(distance_prev, distance_next)
			# Find the points at dmin distance from the summit
			point_prev = Toolkit.find_point_at_distance(contour, s_idx, dmin, direction='backward')
			point_next = Toolkit.find_point_at_distance(contour, s_idx, dmin, direction='forward')
	
			summit_point = contour[s_idx]
			theta = Toolkit.compute_internal_angle(point_prev, summit_point, point_next)
			# Compute the inscribed circle
			A_corner, V_corner, B_corner = point_prev, summit_point, point_next
			radius_corner, center_corner = Toolkit.compute_inscribed_circle_triangle(A_corner, V_corner, B_corner)
			diameter_corner = 2 * radius_corner
			# Compute the farthest point from this corner
			distances_corner = np.linalg.norm(contour - summit_point, axis=1)
			max_distance_idx_corner = np.argmax(distances_corner)
			farthest_point_corner = contour[max_distance_idx_corner]
			max_distance_corner = distances_corner[max_distance_idx_corner]
			# Compute the distance to the closest inscribed circle center
			if len(ins_x) == 1:  # Only one point
				center_to_summit = np.linalg.norm(np.array(summit_point) - np.array([ins_x[0], ins_y[0]]))
			else:  # Create a 2D array of points (ins_x, ins_y)
				points = np.array([ins_x, ins_y]).T
				# Calculate the distance to summit_point for each point
				distances = np.linalg.norm(points - np.array(summit_point), axis=1)
				center_to_summit = np.min(distances)
	
			corner_info = {
				'corner_index': s_idx,
				'corner_point': summit_point,
				'center_to_corner': center_to_summit,
				'angle_degrees': np.degrees(theta),
				'inscribed_circle_diameter': diameter_corner,
				'distance_to_farthest_point': max_distance_corner
			}
			corner_properties.append(corner_info)
		# Print all the informations per corner.
		if display_plot == True:
			for idx, corner in enumerate(corner_properties):
				print(f"\nCorner {idx + 1}:")
				print(f"  Corner Index: {corner['corner_index']}")
				print(f"  Corner Point: {corner['corner_point']}")
				print(f"  Distance to center: {corner['center_to_corner']}")
				print(f"  Angle (degrees): {corner['angle_degrees']:.2f}")
				print(f"  Inscribed Circle Diameter: {corner['inscribed_circle_diameter']:.2f}")
				print(f"  Distance to Farthest Point: {corner['distance_to_farthest_point']:.2f}")
		# Prepare the output dictionary
		output = {
			'max_feret': max_feret,
			'max_feret_angle': max_feret_angle,
			'perpendicular_feret': perpendicular_feret,
			'perpendicular_feret_angle': perpendicular_angle,
			'diameter_sharpest_corner': diameter_sharpest_corner,
			'longest_distance': max_distance,
			'corners': corner_properties
		}
		# Visualization of all measurments made.
		if display_plot == True:
			# Convert the image to RGB to allow drawing with colors
			output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			# Draw the contour
			contour_int = contour.astype(np.int32)
			contour_to_draw = contour_int.reshape((-1, 1, 2))
			cv2.drawContours(output_image, [contour_to_draw], -1, (255, 0, 0), 1)
			# Draw the sharpest vertex
			cv2.circle(output_image, tuple(sharpest_vertex.astype(int)), 5, (0, 0, 255), -1)
			# Draw the inscribed circle at the sharpest corner
			if radius > 0:
				center_int = tuple(np.round(center).astype(int))
				radius_int = int(np.round(radius))
				cv2.circle(output_image, center_int, radius_int, (0, 255, 0), 2)
			else:
				print("Radius is zero or negative. Circle will not be drawn.")
			# Draw the triangle formed by the three points at the sharpest corner
			cv2.line(output_image, tuple(A.astype(int)), tuple(V.astype(int)), (0, 255, 255), 1)
			cv2.line(output_image, tuple(V.astype(int)), tuple(B.astype(int)), (0, 255, 255), 1)
			cv2.line(output_image, tuple(B.astype(int)), tuple(A.astype(int)), (0, 255, 255), 1)
			# Draw the farthest point from the sharpest corner
			cv2.circle(output_image, tuple(farthest_point.astype(int)), 5, (255, 0, 255), -1)
			# Draw the Maximum Feret Diameter
			cv2.line(output_image, tuple(point1_max.astype(int)), tuple(point2_max.astype(int)), (0, 255, 0), 2)
			cv2.putText(output_image, 'Max Feret', tuple(point1_max.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
			# Draw the Perpendicular Feret Diameter
			cv2.line(output_image, tuple(point1_perp.astype(int)), tuple(point2_perp.astype(int)), (0, 165, 255), 2)
			cv2.putText(output_image, 'Perp Feret', tuple(point1_perp.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
			# Draw the Axis Through the Sharpest Corner
			# Define the axis as the line connecting the sharpest corner and the farthest point
			axis_point1 = sharpest_vertex.astype(int)
			axis_point2 = farthest_point.astype(int)
			cv2.line(output_image, tuple(axis_point1), tuple(axis_point2), (255, 255, 0), 2)
			cv2.putText(output_image, 'Axis Sharpest', tuple(axis_point1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
			# Highlight all corners
			for corner in corner_properties:
				cv2.circle(output_image, tuple(corner['corner_point'].astype(int)), 5, (0, 0, 255), -1)
			# Use plt to display the image
			plt.figure(figsize=(10, 10))
			plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
			plt.title('Object Contour and Calculated Points')
			plt.axis('off')
			plt.show()
		return output

	@staticmethod
	def get_centroid(contour):
		"""
		From an open CV contour get its centroid using moments.
		
		Parameters:
		- contour (list): open CV contour, basically an ordered list of list of point coordinates [[x1,y1], ... ,[xn,yn]].
		
		Returns:
		- centroid (list): The centroid position in pixel coordinates (int).
		"""
		if cv2.contourArea(contour) > 0:
			M = cv2.moments(contour)
			cx = int(M['m10'] / M['m00'])
			cy = int(M['m01'] / M['m00'])
			return [cx, cy]
		else:
			print('An error occured (get_centroid): no contour found.')
			return None
	
	@staticmethod
	def angle(a, v, b):
		"""
		Computes the angle at vertex v formed by points a, v, and b.

		Parameters:
		- a (ndarray): coordinates of point a as a 2-element array.
		- v (ndarray): coordinates of vertex point v as a 2-element array.
		- b (ndarray): coordinates of point b as a 2-element array.

		Returns:
		- angle (float): the angle in radians between vectors va and vb.
		"""
		va = a - v
		vb = b - v
		dot_product = np.dot(va, vb)
		magnitude_product = np.linalg.norm(va) * np.linalg.norm(vb)
		cos_theta = dot_product / magnitude_product
		return np.arccos(np.clip(cos_theta, -1.0, 1.0))
	
	@staticmethod
	def wavelet_feature_maps(image, num_levels=3, wavelet_name='bior4.4'):
		"""
		Computes the feature maps of a 2D fast wavelet transform up to the specified order.
	
		Parameters:
		- image (ndarray): 2D numpy array representing the image.
		- num_levels (int): the number of decomposition levels (orders).
		- wavelet_name (str): the name of the wavelet to use.
	
		Returns:
		- coeffs_list: list of length num_levels, where each element is a list containing the 4 feature maps at that level: [cA, cH, cV, cD].
		"""
		coeffs_list = []
		cA = image
		for level in range(num_levels):
			# Perform single-level 2D Discrete Wavelet Transform
			cA, (cH, cV, cD) = pywt.dwt2(cA, wavelet_name)
			# Store the coefficients for this level
			coeffs_list.append([cA, cH, cV, cD])
		return coeffs_list
	
	@staticmethod
	def cartesian_to_polar(contour, centroid):
		"""
		Converts Cartesian coordinates of a contour to polar coordinates relative to a given centroid.
	
		Parameters:
		- contour (ndarray): Nx2 array representing the x, y coordinates of the contour points.
		- centroid (tuple): (cx, cy) coordinates of the centroid.
	
		Returns:
		- rho (ndarray): radial distances of each contour point from the centroid.
		- phi (ndarray): angular positions (in radians) of each contour point relative to the centroid.
		"""
		x = contour[:, 0] - centroid[0]
		y = contour[:, 1] - centroid[1]
		rho = np.sqrt(x**2 + y**2)
		phi = np.arctan2(y, x)
		return rho, phi
	
	@staticmethod
	def selected_points(arr, n):
		"""
		Returns every nth 2D point from a given array.
	
		Parameters:
		- arr: the input array of shape (415, 2).
		- n (int): the step size.
	
		Returns:
		- A subset of the input array with every nth point.
		"""
		if n <= 0:
			raise ValueError("The step size n should be a positive integer.")
		# Return the array with steps, but ensure the last point is unique
		return np.vstack([arr[::n], arr[-1]]) if arr[-1].tolist() not in arr[::n].tolist() else arr[::n]
	
	@staticmethod
	def compute_internal_angle(a, v, b):
		"""
		Calculates the internal angle at point v formed by points a, v, and b.
	
		Parameters:
		- a (numpy.ndarray): coordinates of point a.
		- v (numpy.ndarray): coordinates of vertex point v.
		- b (numpy.ndarray): coordinates of point b.
	
		Returns:
		- angle (float): internal angle in radians between 0 and 2π.
		"""
		va = a - v
		vb = b - v
		# Compute the angle between va and vb using arctan2
		cross = va[0]*vb[1] - va[1]*vb[0]
		dot = np.dot(va, vb)
		angle = np.arctan2(cross, dot)
		if angle < 0:
			angle += 2 * np.pi  # Normalize angle to be between 0 and 2π
		return angle
	
	@staticmethod
	def compute_inscribed_circle_triangle(A, B, C):
		"""
		Computes the radius and incenter of the inscribed circle of a triangle. Used to compute the diameter
		for roudness measurments at each vertex of the particle.
	
		Parameters:
		- A, B, C (numpy.ndarray): coordinates of the triangle's vertices.
	
		Returns:
		- radius (float): inscribed circle radius.
		- incenter (numpy.ndarray): inscribed circle center coords.
		"""
		# Compute side lengths
		a = np.linalg.norm(B - C)
		b = np.linalg.norm(A - C)
		c = np.linalg.norm(A - B)
		# Compute the perimeter
		perimeter = a + b + c
		if perimeter == 0:
			return 0, np.array([0, 0])
		# Compute the incenter
		incenter = (a * A + b * B + c * C) / perimeter
		# Compute the area of the triangle using Heron's formula
		s = perimeter / 2
		area_term = s * (s - a) * (s - b) * (s - c)
		if area_term <= 0:
			radius = 0
		else:
			area = np.sqrt(area_term)
			radius = area / s
		return radius, incenter
	
	@staticmethod
	def compute_feret_diameters(contour, angles=np.arange(0, 180, 1)):
		"""
		Computes Feret diameters for a given contour at specified angles.
	
		Parameters:
		- contour (numpy.ndarray): array of contour points (Nx2).
		- angles (numpy.ndarray): array of angles in degrees at which to compute Feret diameters.
	
		Returns:
		- feret_diameters (dict): a dictionary with angles as keys and corresponding Feret diameters as values.
		"""
		feret_diameters = {}
		for angle in angles:
			# Convert angle to radians
			theta = np.deg2rad(angle)
			# Rotation matrix (clockwise)
			rotation_matrix = np.array([
				[np.cos(theta), np.sin(theta)],
				[-np.sin(theta), np.cos(theta)]
			])
			# Rotate contour
			rotated_contour = contour @ rotation_matrix
			# Compute Feret diameter: max_x - min_x
			min_x = np.min(rotated_contour[:, 0])
			max_x = np.max(rotated_contour[:, 0])
			feret = max_x - min_x
			feret_diameters[angle] = feret
		return feret_diameters
	
	@staticmethod
	def find_point_at_distance(contour, start_idx, distance, direction='forward'):
		"""
		Finds a point at a specified distance along the contour from a starting index.

		Parameters:
		- contour (ndarray): array of shape (N, 2) with contour points.
		- start_idx (int): starting index in the contour.
		- distance (float): distance to travel along the contour.
		- direction (str): direction to travel, either 'forward' or 'backward'.

		Returns:
		- point (ndarray): coordinates of the point at the specified distance.
		"""
		n = len(contour)
		dist = 0
		idx = start_idx
		if direction == 'forward':
			while dist < distance:
				next_idx = (idx + 1) % n
				segment_length = np.linalg.norm(contour[next_idx] - contour[idx])
				if dist + segment_length >= distance:
					ratio = (distance - dist) / segment_length
					point = contour[idx] + ratio * (contour[next_idx] - contour[idx])
					return point
				dist += segment_length
				idx = next_idx
				if idx == start_idx:
					break  # Prevent infinite loop
		elif direction == 'backward':
			while dist < distance:
				prev_idx = (idx - 1) % n
				segment_length = np.linalg.norm(contour[idx] - contour[prev_idx])
				if dist + segment_length >= distance:
					ratio = (distance - dist) / segment_length
					point = contour[idx] + ratio * (contour[prev_idx] - contour[idx])
					return point
				dist += segment_length
				idx = prev_idx
				if idx == start_idx:
					break  # Prevent infinite loop
		return contour[idx]
	
	@staticmethod
	def distance_along_contour(contour, idx1, idx2):
		"""
		Computes the distance along the contour from index idx1 to idx2 in the forward direction.

		Parameters:
		- contour (ndarray): array of shape (N, 2) with contour points.
		- idx1 (int): starting index.
		- idx2 (int): ending index.

		Returns:
		- distance (float): total distance along the contour from idx1 to idx2.
		"""
		if idx1 == idx2:
			return 0
		dist = 0
		n = len(contour)
		i = idx1
		while True:
			next_i = (i + 1) % n
			dist += np.linalg.norm(contour[next_i] - contour[i])
			i = next_i
			if i == idx2 or i == idx1:
				break
		return dist
	
	@staticmethod
	def distance_along_contour_reverse(contour, idx1, idx2):
		"""
		Computes the distance along the contour from index idx1 to idx2 in the reverse direction.

		Parameters:
		- contour (ndarray): array of shape (N, 2) with contour points.
		- idx1 (int): starting index.
		- idx2 (int): ending index.

		Returns:
		- distance (float): total distance along the contour from idx1 to idx2 in reverse.
		"""
		if idx1 == idx2:
			return 0
		dist = 0
		n = len(contour)
		i = idx1
		while True:
			prev_i = (i - 1) % n
			dist += np.linalg.norm(contour[i] - contour[prev_i])
			i = prev_i
			if i == idx2 or i == idx1:
				break
		return dist
	
	@staticmethod
	def compute_statistics(data):
		"""
		Computes energy, entropy, mean, standard deviation, and contrast of the input data.
	
		Parameters:
		- data (ndarray): numpy array of coefficients.
	
		Returns:
		- stats (dict): dictionary containing the computed statistics.
		"""
		# Flatten data to 1D array
		data = data.flatten()
		# Energy: Sum of squares of the coefficients
		energy = np.sum(data ** 2)
		# Histogram for entropy calculation
		histogram, _ = np.histogram(data, bins=256, density=True)
		histogram += np.finfo(float).eps  # Avoid log(0)
		probabilities = histogram / np.sum(histogram)
		# Entropy: Measure of randomness
		entropy = -np.sum(probabilities * np.log2(probabilities))
		# Mean of the coefficients
		mean = np.mean(data)
		# Standard deviation of the coefficients
		std_dev = np.std(data)
		# Contrast: Variance of the coefficients
		contrast = std_dev ** 2
		stats = {
			'energy': energy,
			'entropy': entropy,
			'mean': mean,
			'std_dev': std_dev,
			'contrast': contrast
		}
		return stats
	
	@staticmethod
	def mid_points(binary_object, contour):
		"""
		Computes midpoints for each contour pixel where the pixel side faces a background pixel.

		Parameters:
		- binary_object (ndarray): a binary image where the object is white (255) and background is black (0).
		- contour (ndarray): array of shape (N, 2) with ordered contour pixel coordinates.
		- tol (float): tolerance for removing duplicate midpoints.

		Returns:
		- unique_candidates (ndarray): array of shape (M, 2) with unique midpoint coordinates in sub-pixel precision.
		"""
		mid_points = []
		height, width = binary_object.shape
		for pixel in contour:
			j, i = int(pixel[0]), int(pixel[1]) #opencv
			# List of potential mid points and their corresponding neighbor pixels
			potential_points = [((i + 0.5, j + 1), (i, j + 1)),((i, j + 0.5), (i - 1, j)),((i + 1, j + 0.5), (i + 1, j)),((i + 0.5 , j), (i,j - 1))]
			h=0
			test=[]
			neighboor=[]
			for mid_point, neighbor in potential_points:
				neighbor_i, neighbor_j = neighbor
				# Check if the neighbor pixel is within the image and is a background pixel
				h=0
				print(mid_point,neighbor)
				if neighbor_i<height and neighbor_j<width :
				  neighboor.append([(neighbor_i, neighbor_j),binary_object[neighbor_i, neighbor_j]])
				  if (0 <= neighbor_i < height and 0 <= neighbor_j < width and binary_object[neighbor_i, neighbor_j] ==0):
					  mid_points.append(mid_point)
					  test.append(mid_point)
					  h=h+1
		return np.array(mid_points)
	
	@staticmethod
	def uniform_important_sampling(contour, epsilon,num_points=0.2,distance_overlap=0.01):
		"""
		Uniformly samples key points from a contour by combining vertices obtained via the Ramer–Douglas–Peucker (RDP)
		algorithm with uniformly spaced points along the contour. The method filters out points that are too close to
		the approximated vertices (using the specified distance threshold) and returns a set of final points sorted
		by their angular orientation relative to the contour’s centroid.

		Parameters:
		- contour (ndarray): array of contour points of shape (N, 2) representing the object boundary.
		- epsilon (float): approximation accuracy parameter for the RDP algorithm.
		- num_points (float or int): number or ratio of points to uniformly sample along the contour.
		- distance_overlap (float): threshold distance to filter out points that are too close to any approximated vertex.

		Returns:
		- final_points (ndarray): an array of sampled contour points, sorted in increasing order of their polar angle 
		  about the centroid.
		"""
		approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()
		# Calculate cumulative distance along the contour
		distances = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
		distances = np.insert(distances, 0, 0)
		# Sample points uniformly based on the cumulative distance
		nb_points_contour = len(contour)
		if num_points <= 1:
		  interval=int(1/num_points)
	
		else:
		  interval = int(nb_points_contour / num_points)
	
		indices = np.arange(0, nb_points_contour, interval)
		# Filter out indices that reference points too close to any point in approx
		mask = np.ones(len(indices), dtype=bool)
		for idx in indices:
			for a_point in approx:
				if np.linalg.norm(contour[idx] - a_point) < 0.01:
					mask[np.where(indices == idx)[0]] = False
					break
		filtered_indices = indices[mask]
		sampled_points = contour[filtered_indices]
		# Combine the approx points and the uniformly sampled points
		final_points = np.vstack((approx, sampled_points))
		# Sort final_points based on their angles with respect to the centroid
		centroid = np.mean(final_points, axis=0)
		angles = np.arctan2(final_points[:, 1] - centroid[1], final_points[:, 0] - centroid[0])
		sorted_indices = np.argsort(angles)
		final_points = final_points[sorted_indices]
		return final_points
	
	@staticmethod
	def line_segment_intersection(p1, p2, p3, p4, eps=1e-12):
		"""
		Calculates the intersection point of two line segments defined by endpoints p1, p2 and p3, p4 using a parametric approach.
		
		Parameters:
		- p1 (tuple): first endpoint (x, y) of the first line segment.
		- p2 (tuple): second endpoint (x, y) of the first line segment.
		- p3 (tuple): first endpoint (x, y) of the second line segment.
		- p4 (tuple): second endpoint (x, y) of the second line segment.
		- eps (float): tolerance to determine if the segments are nearly parallel (default 1e-12).
		
		Returns:
		- tuple or None: a tuple (xi, yi, t1, t2) where (xi, yi) is the intersection point and t1, t2 are the parametric values 
		  along each segment if an intersection exists within the segments; otherwise, returns None.
    	"""
		x1, y1 = p1
		x2, y2 = p2
		x3, y3 = p3
		x4, y4 = p4
		denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)

		if abs(denom) < eps:
			return None  # Parallel or almost parallel

		t1 = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom
		t2 = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom

		if 0 - eps <= t1 <= 1 + eps and 0 - eps <= t2 <= 1 + eps:
			xi = x1 + t1*(x2 - x1)
			yi = y1 + t1*(y2 - y1)
			return (xi, yi, t1, t2)
		return None

	@staticmethod
	def polygon_midpoints(binary_object, contour, tol=1e-6):
		"""
		Vectorized computation of midpoints for each contour pixel where the pixel side faces a background pixel.
		Each contour pixel (assumed to be given as [x, y], with (0,0) at the top-left) contributes four candidate midpoints.
		
		Parameters:
		- binary_object (ndarray): a binary image where the object of interest is white (255) and background is black (0).
		- contour (list): array of shape (N,2) with ordered contour pixel coordinates.
		- tol (float): tolerance used when removing duplicate midpoints.
		
		Returns:
		- unique_candidates: array of shape (M,2) of unique candidate midpoints (sub-pixel coordinates), 
		  in the order in which they first appear along the contour.
		"""
		# Ensure contour is an integer array.
		contour = np.asarray(contour, dtype=int)
		N = contour.shape[0]
		height, width = binary_object.shape
		# Extract x and y coordinates (note: x is column, y is row)
		x = contour[:, 0]  # shape (N,)
		y = contour[:, 1]  # shape (N,)
		# Candidate midpoints for each pixel, shape: (N, 4, 2)
		# For a pixel at (x,y), candidates are:
		# - Right edge: (x + 0.5, y)
		# - Bottom edge: (x, y + 0.5)
		# - Left edge: (x - 0.5, y)
		# - Top edge: (x, y - 0.5)
		mid_candidates = np.empty((N, 4, 2), dtype=float)
		mid_candidates[:, 0, 0] = x + 0.5  # Right edge: (x+0.5, y)
		mid_candidates[:, 0, 1] = y
		mid_candidates[:, 1, 0] = x         # Bottom edge: (x, y+0.5)
		mid_candidates[:, 1, 1] = y + 0.5
		mid_candidates[:, 2, 0] = x - 0.5     # Left edge: (x-0.5, y)
		mid_candidates[:, 2, 1] = y
		mid_candidates[:, 3, 0] = x         # Top edge: (x, y-0.5)
		mid_candidates[:, 3, 1] = y - 0.5
		# Compute the corresponding neighbor coordinates (which lie just outside the object)
		# These are integer coordinates (since the pixel is assumed to start at (x, y)):
		# - Right neighbor: (x+1, y)
		# - Bottom neighbor: (x, y+1)
		# - Left neighbor: (x-1, y)
		# - Top neighbor: (x, y-1)
		neighbors = np.empty((N, 4, 2), dtype=int)
		neighbors[:, 0, 0] = x + 1  # Right neighbor
		neighbors[:, 0, 1] = y
		neighbors[:, 1, 0] = x      # Bottom neighbor
		neighbors[:, 1, 1] = y + 1
		neighbors[:, 2, 0] = x - 1  # Left neighbor
		neighbors[:, 2, 1] = y
		neighbors[:, 3, 0] = x      # Top neighbor
		neighbors[:, 3, 1] = y - 1
		# Check bounds for neighbors:
		neighbor_x = neighbors[:, :, 0]
		neighbor_y = neighbors[:, :, 1]
		in_bounds = (neighbor_x >= 0) & (neighbor_x < width) & (neighbor_y >= 0) & (neighbor_y < height)
		# For neighbors that are in bounds, check if they are background (i.e. pixel value 0).
		# (binary_object is indexed as [row, col] or [y, x])
		valid = np.zeros((N, 4), dtype=bool)
		valid[in_bounds] = (binary_object[neighbor_y[in_bounds], neighbor_x[in_bounds]] == 0)
		# Get candidate midpoints for which valid is True.
		valid_candidates = mid_candidates[valid]  # shape (M, 2) for some M <= 4*N
		# Order-preserving duplicate removal.
		decimals = int(-np.log10(tol)) if tol > 0 else 6
		rounded = np.round(valid_candidates, decimals=decimals)
		seen = {}
		result = []
		for i, row in enumerate(rounded):
			tup = tuple(row)
			if tup not in seen:
				seen[tup] = i
				result.append(valid_candidates[i])
		unique_candidates = np.array(result)
		return unique_candidates

	@staticmethod
	def find_bspline_polygon_intersections(bspline_pts, polygon, closed=True, near_tol=0.6):
		"""
		Finds intersection points between a B-spline curve and a polygon by checking segments.

		Parameters:
		- bspline_pts (ndarray): array of shape (N, 2) representing the B-spline curve points.
		- polygon (ndarray): array of shape (M, 2) with polygon vertices.
		- closed (bool): indicates whether the polygon is closed (default True).

		Returns:
		- intersection_points (ndarray): array of shape (K, 2) with intersection coordinates sorted by B-spline parameter.
		"""
		eps = 1e-12
		# Create polygon edges.
		if closed:
			poly_edges = np.vstack([polygon, polygon[0]])
		else:
			poly_edges = polygon
		v1 = poly_edges[:-1]  # shape (Np, 2)
		v2 = poly_edges[1:]   # shape (Np, 2)
		Np = v1.shape[0]
		# Bspline segments: each segment from bspline_pts[i] to bspline_pts[i+1]
		M = bspline_pts.shape[0]
		if M < 2:
			return np.empty((0, 2))
		num_seg = M - 1
		p1_seg = bspline_pts[:-1]  # shape (num_seg, 2)
		p2_seg = bspline_pts[1:]   # shape (num_seg, 2)
		# Repeat the bspline segments and polygon edges so that both have shape (num_seg, Np, 2)
		p1_exp = np.repeat(p1_seg[:, np.newaxis, :], Np, axis=1)  # (num_seg, Np, 2)
		p2_exp = np.repeat(p2_seg[:, np.newaxis, :], Np, axis=1)  # (num_seg, Np, 2)
		v1_exp = np.repeat(v1[np.newaxis, :, :], num_seg, axis=0)  # (num_seg, Np, 2)
		v2_exp = np.repeat(v2[np.newaxis, :, :], num_seg, axis=0)  # (num_seg, Np, 2)
		# Extract coordinates.
		x1 = p1_exp[..., 0]
		y1 = p1_exp[..., 1]
		x2 = p2_exp[..., 0]
		y2 = p2_exp[..., 1]
		x3 = v1_exp[..., 0]
		y3 = v1_exp[..., 1]
		x4 = v2_exp[..., 0]
		y4 = v2_exp[..., 1]
		# Compute denominator for strict intersection.
		denom = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
		non_zero = np.abs(denom) > eps
		
		t1 = np.where(non_zero, ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / denom, -1)
		t2 = np.where(non_zero, ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / denom, -1)
		
		valid_strict = non_zero & (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
		xi = x1 + t1*(x2 - x1)
		yi = y1 + t1*(y2 - y1)
		
		seg_idx = np.arange(num_seg)[:, np.newaxis]  # shape (num_seg, 1)
		global_param = seg_idx + t1  # shape (num_seg, Np)
		# Gather strict intersections.
		xi_valid = xi[valid_strict]
		yi_valid = yi[valid_strict]
		gp_valid = global_param[valid_strict]
		# --- Near-detection ---
		# Compute vectorized quantities for closest distance.
		d1 = p2_exp - p1_exp    # (num_seg, Np, 2)
		d2 = v2_exp - v1_exp    # (num_seg, Np, 2)
		r = p1_exp - v1_exp     # (num_seg, Np, 2)
		a = np.sum(d1 * d1, axis=-1)  # (num_seg, Np)
		b = np.sum(d1 * d2, axis=-1)
		c = np.sum(d2 * d2, axis=-1)
		d_val = np.sum(d1 * r, axis=-1)
		e_val = np.sum(d2 * r, axis=-1)
		denom2 = a * c - b * b
		valid_denom = denom2 > eps
		# Flatten arrays.
		a_flat = a.ravel()
		b_flat = b.ravel()
		c_flat = c.ravel()
		d_flat = d_val.ravel()
		e_flat = e_val.ravel()
		denom2_flat = denom2.ravel()
		valid_denom_flat = valid_denom.ravel()
		
		t1_near_flat = np.zeros_like(a_flat)
		t2_near_flat = np.zeros_like(a_flat)
		t1_near_flat[valid_denom_flat] = (b_flat[valid_denom_flat] * e_flat[valid_denom_flat] -
										  c_flat[valid_denom_flat] * d_flat[valid_denom_flat]) / denom2_flat[valid_denom_flat]
		t2_near_flat[valid_denom_flat] = (a_flat[valid_denom_flat] * e_flat[valid_denom_flat] -
										  b_flat[valid_denom_flat] * d_flat[valid_denom_flat]) / denom2_flat[valid_denom_flat]
		t1_near = t1_near_flat.reshape(a.shape)
		t2_near = t2_near_flat.reshape(a.shape)
		t1_near = np.clip(t1_near, 0, 1)
		t2_near = np.clip(t2_near, 0, 1)
		
		cp1 = p1_exp + t1_near[..., np.newaxis] * d1
		cp2 = v1_exp + t2_near[..., np.newaxis] * d2
		dist = np.linalg.norm(cp1 - cp2, axis=-1)
		
		near_mask = (~valid_strict) & (dist < near_tol)
		approx_x = (cp1[..., 0] + cp2[..., 0]) / 2
		approx_y = (cp1[..., 1] + cp2[..., 1]) / 2
		gp_near = seg_idx + t1_near  # use t1_near for global parameter
		
		approx_x_valid = approx_x[near_mask]
		approx_y_valid = approx_y[near_mask]
		gp_near_valid = gp_near[near_mask]
		# Combine strict and near-detected intersections.
		if xi_valid.size == 0 and approx_x_valid.size == 0:
			return np.empty((0, 2))
		all_x = np.concatenate([xi_valid, approx_x_valid])
		all_y = np.concatenate([yi_valid, approx_y_valid])
		all_gp = np.concatenate([gp_valid, gp_near_valid])
		
		sort_idx = np.argsort(all_gp)
		intersection_points = np.column_stack((all_x[sort_idx], all_y[sort_idx]))
		return intersection_points
	
	@staticmethod
	def downsample_array(arr, max_points=500):
		"""
		Uniformly downsamples an array to a maximum number of points if it exceeds that limit.

		Parameters:
		- arr (ndarray): input array of shape (N, D) to downsample.
		- max_points (int): maximum number of points to retain (default 500).

		Returns:
		- downsampled (ndarray): array of shape (M, D) where M <= max_points.
		"""
		n = len(arr)
		if n <= max_points:
			return arr
		# Build a uniform sample of indices
		indices = np.round(np.linspace(0, n - 1, max_points)).astype(int)
		return arr[indices]
	
	@staticmethod
	def bspline_fit(points, degree=3, num_samples=200, s=0.0, closed=True):
		"""
		Fits a B-spline to 2D points and samples points along the curve.

		Parameters:
		- points (ndarray): array of shape (N, 2) with x, y coordinates.
		- degree (int): degree of the B-spline (default 3).
		- num_samples (int): number of points to sample on the B-spline (default 200).
		- s (float): smoothing factor; 0 means interpolate through points (default 0.0).
		- closed (bool): indicates if the B-spline is closed (default True).

		Returns:
		- bspline_pts (ndarray): array of shape (num_samples, 2) with sampled B-spline points.
		- tck (tuple): B-spline representation (knots, coefficients, degree).
    	"""
		x = points[:, 0]
		y = points[:, 1]
	
		# per=1 => treat as closed if data is roughly circular
		per_flag = 1 if closed else 0
	
		tck, _ = splprep([x, y], k=degree, s=s, per=per_flag)
		u_new = np.linspace(0, 1, num_samples)
		out = splev(u_new, tck)
		bspline_pts = np.vstack(out).T
		return bspline_pts, tck
	
	@staticmethod
	def find_bspline_polygon_intersections(bspline_pts, polygon, closed=True):
		"""
		Finds the intersection points between a B-spline curve and a polygon by checking each segment of the B-spline
		against all edges of the polygon.
		
		Parameters:
		- bspline_pts (ndarray): an array of points representing the B-spline curve.
		- polygon (ndarray): an array of vertices representing the polygon.
		- closed (bool): indicates whether the polygon is closed (default is True).
		
		Returns:
		- intersection_points (ndarray): an array of intersection points (each as [x, y]) sorted by their global parameter along the B-spline.
		  Returns an empty array if no intersections are found.
		"""
		poly_edges = []
		Np = len(polygon)
		for i in range(Np):
			j = (i + 1) % Np if closed else i + 1
			if j < Np:
				poly_edges.append((polygon[i], polygon[j]))
	
		intersection_data = []
		M = len(bspline_pts)
		for i in range(M - 1):
			p1 = tuple(bspline_pts[i])
			p2 = tuple(bspline_pts[i + 1])
	
			for (v1, v2) in poly_edges:
				result = Toolkit.line_segment_intersection(p1, p2, tuple(v1), tuple(v2))
				if result is not None:
					xi, yi, t1, _ = result
					t_global = i + t1
					intersection_data.append((xi, yi, t_global))
	
		if len(intersection_data) == 0:
			return np.array([])
	
		intersection_data.sort(key=lambda x: x[2])
		intersection_points = np.array([[x, y] for (x, y, _) in intersection_data], dtype=float)
		return intersection_points
	
	@staticmethod
	def downsample_points(pts, step=5):
		"""
		Downsamples an array of points by selecting every 'step'-th point, ensuring that the last point is always included.
		
		Parameters:
		- pts (array-like): an array of points.
		- step (int): the interval at which points are sampled (default is 5).
		
		Returns:
		- subsampled (ndarray): the downsampled array of points.
		"""
		pts = np.array(pts)
		if len(pts) == 0:
			return pts
		subsampled = pts[::step]
		# ensure last point included
		if not np.array_equal(subsampled[-1], pts[-1]):
			subsampled = np.vstack([subsampled, pts[-1]])
		return subsampled
	
	@staticmethod
	def true_perp_distance_bspline(pt, tck):
		"""
		Calculates the minimum (true perpendicular) distance from a given point to a B-spline curve represented by its spline parameters.
		
		Parameters:
		- pt (tuple or array): the point (x, y) for which the distance is computed.
		- tck (tuple): the B-spline representation (knots, coefficients, degree) returned by splprep.
		
		Returns:
		- min_dist (float): the minimum perpendicular distance from the point to the B-spline curve.
		"""
		def dist_sq(u):
			x_u, y_u = splev(u, tck)
			return (x_u - pt[0])**2 + (y_u - pt[1])**2
	
		res = minimize_scalar(dist_sq, bounds=(0, 1), method='bounded')
		min_dist = np.sqrt(res.fun)
		return min_dist
###############################################################################################################