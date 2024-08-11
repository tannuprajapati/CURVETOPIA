import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))
    colours = ['black']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

def detect_line(points, threshold=0.01):
    x = points[:, 0]
    y = points[:, 1]
   
    model = RANSACRegressor()
    model.fit(x.reshape(-1, 1), y)
    m = model.estimator_.coef_[0]
    c = model.estimator_.intercept_
   
    residuals = y - (m * x + c)
    mse = np.mean(residuals**2)
   
    return mse < threshold, (m, c)

def detect_circle(points, threshold=0.01):
    x, y = points[:, 0], points[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 3)
    Svv = np.sum(v ** 3)
    Suuv = np.sum(u ** 3 * v)
    Suvv = np.sum(u * v ** 3)
    Suuu = np.sum(u ** 4)
    Svvv = np.sum(v ** 4)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)
    xc, yc = x_m + uc, y_m + vc
    R = np.sqrt(uc ** 2 + vc ** 2 + (Suu + Svv) / len(x))
    residuals = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R
    mse = np.mean(residuals ** 2)
    1234
   
    return mse < threshold, (xc, yc, R)

def detect_rectangle(points, threshold=0.05):
    if len(points) < 4:
        return False, points

    # Check if points form a rectangle
    edges = np.roll(points, -1, axis=0) - points
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angle_diffs = np.abs(np.diff(np.concatenate([angles, [angles[0]]])) % np.pi)
    right_angles = np.sum(np.isclose(angle_diffs, np.pi / 2, atol=threshold))

    if right_angles >= 4:
        # Regularize the rectangle
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        regularized_points = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min]
        ])
        return True, regularized_points
    return False, points

def regularize_shape(points):
    is_line, line_params = detect_line(points)
    if is_line:
        return "line", line_params

    is_circle, circle_params = detect_circle(points)
    if is_circle:
        return "circle", circle_params

    is_rectangle, rect_params = detect_rectangle(points)
    if is_rectangle:
        return "rectangle", rect_params

    return "irregular", points

def process_shapes(path_XYs):
    regularized_shapes = []
    for shape in path_XYs:
        regularized_shape = []
        for path in shape:
            shape_type, params = regularize_shape(path)
            regularized_shape.append((shape_type, params))
        regularized_shapes.append(regularized_shape)
    return regularized_shapes

def plot_regularized(regularized_shapes):
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))
    colours = ['black']
    for i, shape in enumerate(regularized_shapes):
        c = colours[i % len(colours)]
        for shape_type, params in shape:
            if shape_type == "line":
                m, b = params
                x = np.array([0, 250])  # Assuming some range for the line
                y = m * x + b
                ax.plot(x, y, c=c)
            elif shape_type == "circle":
                xc, yc, r = params
                circle = plt.Circle((xc, yc), r, fill=False, color=c)
                ax.add_artist(circle)
            elif shape_type == "rectangle":
                rect_points = np.vstack([params, params[0]])  # Close the rectangle loop
                ax.plot(rect_points[:, 0], rect_points[:, 1], c=c)
            else:
                ax.plot(params[:, 0], params[:, 1], c=c)
    ax.set_aspect('equal')
    plt.show()

# Test the implementation
csv_path = "frag0.csv"

path_XYs = read_csv(csv_path)
print("Original shapes:")
plot(path_XYs)

regularized_shapes = process_shapes(path_XYs)
print("Regularized shapes:")
plot_regularized(regularized_shapes)