import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    shapes = []
    for i in np.unique(data[:, 0]):
        shape_data = data[data[:, 0] == i][:, 1:]
        paths = []
        for j in np.unique(shape_data[:, 0]):
            path = shape_data[shape_data[:, 0] == j][:, 1:]
            paths.append(path)
        shapes.append(paths)
    return shapes

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))
    colours = ['black']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

def detect_symmetry(points, num_angles=360):
    def symmetry_error(angle):
        rotated = rotate_points(points, angle)
        flipped = np.copy(rotated)
        flipped[:, 0] = -flipped[:, 0]
        distances = np.min(np.sum((rotated[:, np.newaxis] - flipped) ** 2, axis=2), axis=1)
        return np.mean(distances)

    angles = np.linspace(0, np.pi, num_angles)
    errors = [symmetry_error(angle) for angle in angles]
    best_angle = angles[np.argmin(errors)]
    
    return best_angle, np.min(errors)

def rotate_points(points, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(points - np.mean(points, axis=0), rotation_matrix.T)

def fit_bezier(points, max_degree=3):
    m = len(points)
    for degree in range(min(max_degree, m-1), 0, -1):
        if m > degree:
            try:
                tck, u = splprep([points[:, 0], points[:, 1]], k=degree, s=0)
                return tck
            except:
                continue
    return None

def plot_shape_with_symmetry(points, symmetry_angle, bezier_left, bezier_right):
    plt.figure(figsize=(3, 3))
    plt.plot(points[:, 0], points[:, 1], 'k.', markersize=2)
    
    center = np.mean(points, axis=0)
    sym_line = np.array([[-np.sin(symmetry_angle), np.cos(symmetry_angle)],
                         [np.sin(symmetry_angle), -np.cos(symmetry_angle)]])
    sym_line = center + sym_line * np.max(np.abs(points - center)) * 1.2
    plt.plot(sym_line[:, 0], sym_line[:, 1], 'r--', label='Symmetry Line')
    
    if bezier_left is not None and bezier_right is not None:
        t = np.linspace(0, 1, 100)
        left_curve = splev(t, bezier_left)
        right_curve = splev(t, bezier_right)
        
        plt.plot(left_curve[0], left_curve[1], 'g-', label='Left Bezier')
        plt.plot(right_curve[0], right_curve[1], 'b-', label='Right Bezier')
    
    plt.axis('equal')
    plt.legend()
    plt.show()
    
def process_shape(shape, symmetry_threshold=0.01):
    combined_shape = np.vstack(shape)
    symmetry_angle, symmetry_error = detect_symmetry(combined_shape)
    
    if symmetry_error < symmetry_threshold:
        print(f"Symmetry detected at angle: {np.degrees(symmetry_angle):.2f}°")
        rotated = rotate_points(combined_shape, symmetry_angle)
        left_half = rotated[rotated[:, 0] <= 0]
        right_half = rotated[rotated[:, 0] >= 0]
        right_half[:, 0] = -right_half[:, 0]
        
        bezier_left = fit_bezier(left_half)
        bezier_right = fit_bezier(right_half)
        
        if bezier_left is not None and bezier_right is not None:
            plot_shape_with_symmetry(combined_shape, symmetry_angle, bezier_left, bezier_right)
        else:
            print("Not enough points to fit Bezier curves")
            plot_shape_with_symmetry(combined_shape, symmetry_angle, None, None)
    else:
        print("No symmetry detected")


# Main execution
csv_path = r"frag01_sol.csv"
shapes = read_csv(csv_path)
print("Original shape")
plot(shapes)

for i, shape in enumerate(shapes):
    print(f"Processing shape {i+1}")
    process_shape(shape)
