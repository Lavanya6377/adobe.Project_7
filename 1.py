import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_and_process_image(image_path):
    """Load the image, convert to grayscale, and apply adaptive thresholding."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh, image


def extract_contours(thresh):
    """Extract contours from the thresholded image."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2)
    return points


def reflect_point(point, angle, centroid):
    """Reflect a point across a line at a given angle."""
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    translated_point = point - centroid
    rotated_point = np.array([
        translated_point[0] * cos_theta + translated_point[1] * sin_theta,
        -translated_point[0] * sin_theta + translated_point[1] * cos_theta
    ])
    reflected_point = np.array([
        rotated_point[0],
        -rotated_point[1]
    ])
    final_point = np.array([
        reflected_point[0] * cos_theta - reflected_point[1] * sin_theta,
        reflected_point[0] * sin_theta + reflected_point[1] * cos_theta
    ]) + centroid

    return final_point


def evaluate_symmetry(points, angle, centroid):
    """Evaluate how well the shape is symmetric around a line at the given angle."""
    reflected_points = np.array([reflect_point(point, angle, centroid) for point in points])
    distances = np.linalg.norm(points - reflected_points, axis=1)
    return np.mean(distances)


def find_best_symmetry_line(points, angles, centroid):
    """Find the best line of symmetry by minimizing the distance between original and reflected points."""
    best_angle = None
    min_distance = float('inf')

    for angle in angles:
        distance = evaluate_symmetry(points, angle, centroid)
        print(f"Angle: {angle}, Symmetry Distance: {distance}")  # Debugging statement
        if distance < min_distance:
            min_distance = distance
            best_angle = angle

    return best_angle


def plot_shape_and_lines(image, points, best_angle, centroid):
    """Plot the original shape and the best symmetry line on the image."""
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Plot the original shape
    if points.size > 0:
        plt.plot(points[:, 0], points[:, 1], 'o-', label='Original Shape')
        plt.plot(np.append(points[:, 0], points[0, 0]),
                 np.append(points[:, 1], points[0, 1]), 'r--', label='Closed Shape')

    # Plot the symmetry line
    if best_angle is not None and centroid is not None:
        angle_rad = np.radians(best_angle)
        x_vals = np.linspace(0, image.shape[1], 2)
        y_vals = np.tan(angle_rad) * (x_vals - centroid[0]) + centroid[1]
        plt.plot(x_vals, y_vals, 'g--', label=f'Symmetry Line at {best_angle}°')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Shape and Symmetry Line')
    plt.show()


def main(image_path):
    try:
        thresh, image = load_and_process_image(image_path)
        print("Image loaded and processed successfully.")
        points = extract_contours(thresh)
        print(f"Contours extracted: {points.shape}")

        # Compute centroid
        centroid = np.mean(points, axis=0)
        print(f"Centroid of shape: {centroid}")

        # Determine the number of vertices and symmetry angles
        num_vertices = len(points)
        if num_vertices == 3:
            # For triangles, evaluate possible symmetry lines
            angles = [0, 60, 120]  # Lines of symmetry through medians in an equilateral triangle
        else:
            # For other shapes, use finer angle evaluation
            angles = np.linspace(0, 180, 360)  # Check every degree

        # Find the best angle for symmetry
        best_angle = find_best_symmetry_line(points, angles, centroid)
        print(f'Best symmetry line is at angle: {best_angle}°')

        # Plot the shape and symmetry line
        plot_shape_and_lines(image, points, best_angle, centroid)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    image_path = r"C:\Users\ASUS\Downloads\WhatsApp Image 2024-07-31 at 20.05.38_ed0363b5.jpg"  # Replace with your image path
    main(image_path)
