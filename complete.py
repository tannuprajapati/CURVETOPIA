import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def read_csv(csv_path):
    """Read CSV file and extract shapes."""
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    shapes = []
    for i in np.unique(data[:, 0]):
        shape_data = data[data[:, 0] == i][:, 1:]
        paths = [shape_data[shape_data[:, 0] == j][:, 1:] for j in np.unique(shape_data[:, 0])]
        shapes.append(paths)
    return shapes

def plot_shapes(shapes, regularized=False):
    """Plot shapes with different colors."""
    plt.figure(figsize=(3, 3))
    ax = plt.gca()
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, shape in enumerate(shapes):
        c = colours[i % len(colours)]
        for curve in shape:
            curve = np.array(curve)  # Ensure curve is a NumPy array
            if curve.ndim == 2 and curve.shape[1] == 2:
                ax.plot(curve[:, 0], curve[:, 1], c=c, linewidth=2)
            else:
                print(f"Skipping invalid shape with dimensions: {curve.shape}")
    
    ax.set_aspect('equal')
    plt.show()

def detect_occlusion(curves):
    """Detect occlusions in curves."""
    occlusions = []
    for i, curve1 in enumerate(curves):
        for j, curve2 in enumerate(curves[i+1:], start=i+1):
            if curves_intersect(curve1, curve2):
                occlusions.append((i, j))
    return occlusions

def curves_intersect(curve1, curve2):
    """Check if two curves intersect."""
    for p1 in curve1:
        for p2 in curve2:
            if np.allclose(p1, p2, atol=1e-5):
                return True
    return False

def complete_curve(curve, occlusion_type):
    """Complete a curve based on occlusion type."""
    if occlusion_type == "fully_contained":
        return complete_fully_contained(curve)
    elif occlusion_type == "partially_contained":
        return complete_partially_contained(curve)
    elif occlusion_type == "disconnected":
        return complete_disconnected(curve)
    else:
        return curve

def complete_fully_contained(curve):
    """For fully contained, we assume the curve is complete."""
    return curve

def complete_partially_contained(curve):
    """For partially contained, we use spline interpolation."""
    tck, u = splprep(curve.T, s=0, per=1)
    u_new = np.linspace(0, 1, 1000)
    completed_curve = np.column_stack(splev(u_new, tck))
    return completed_curve

def complete_disconnected(curve_parts):
    """For disconnected, we connect the parts with a smooth curve."""
    combined_curve = np.vstack(curve_parts)
    tck, u = splprep(combined_curve.T, s=0, k=3)
    u_new = np.linspace(0, 1, 1000)
    completed_curve = np.column_stack(splev(u_new, tck))
    return completed_curve

def process_occlusions(shapes):
    """Detect and process occlusions in shapes."""
    completed_shapes = []
    for shape in shapes:
        occlusions = detect_occlusion(shape)
        if not occlusions:
            completed_shapes.append(shape)
        else:
            completed_shape = []
            for i, curve in enumerate(shape):
                occluded = any(i in occ for occ in occlusions)
                if occluded:
                    if len(curve) == 1:
                        occlusion_type = "fully_contained"
                    elif np.allclose(curve[0], curve[-1]):
                        occlusion_type = "partially_contained"
                    else:
                        occlusion_type = "disconnected"
                    completed_curve = complete_curve(curve, occlusion_type)
                    completed_shape.append(completed_curve)
                else:
                    completed_shape.append(curve)
            completed_shapes.append(completed_shape)
    return completed_shapes

def main():
    """Main function to execute the script."""
    csv_path = "isolated.csv"
    shapes = read_csv(csv_path)
    print("Original shapes:")
    plot_shapes(shapes)

    regularized_shapes = process_shapes(shapes)
    print("Regularized shapes:")
    plot_shapes(regularized_shapes)

    completed_shapes = process_occlusions(regularized_shapes)
    print("Completed shapes:")
    plot_shapes(completed_shapes)

if __name__ == "__main__":
    main()
