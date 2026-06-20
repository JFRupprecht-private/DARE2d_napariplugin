from skimage import io
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def draw_line_in_matrix(matrix, start, end, length, num_points, val=1):
    # Generate points along the line using linear interpolation
    vec = end - start
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 0.0:
        vec = vec / vec_norm
        end = start + vec * length

    t = np.linspace(0, 1, num_points)
    line_points = (1 - t)[:, np.newaxis] * start + t[:, np.newaxis] * end

    for point in line_points:
        x, y, z = point
        x_idx = int(x)
        y_idx = int(y)
        z_idx = int(z)
        matrix[x_idx, y_idx, z_idx] = val
    return matrix


def get_quaternion(point1, point2):
    def sort_points_clockwise(points, reference_point):
        # Calculate vectors from reference point to the points
        vectors = points - reference_point

        # Calculate angles of points with respect to the reference point
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Sort the points based on angles in clockwise order
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        return sorted_points
    center = (point1+point2)/2
    point1, point2 = sort_points_clockwise(np.array([point1, point2]), center)
    quat = compute_quaternion(point1, point2)
    return quat


def compute_quaternion(point1, point2):
    vector = point2 - point1
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        # No rotation, return identity quaternion
        return np.array([1, 0, 0, 0])
    axis = vector / norm

    angle = np.arccos(np.dot([1, 0, 0], axis))
    half_angle = angle / 2
    quat = np.array([np.cos(half_angle), np.sin(half_angle) * axis[0],
                    np.sin(half_angle) * axis[1], np.sin(half_angle) * axis[2]])

    return quat.astype(np.float32)


def angle_loss(y_true, y_pred):
    # Φ3(q1, q2) = arccos(|q1 · q2|)
    # or 2 arccos for values in range [0; pi]
    # Normalize the true and predicted quaternions
    y_true_norm = tf.math.l2_normalize(y_true, epsilon=1e-4)
    y_pred_norm = tf.math.l2_normalize(y_pred, epsilon=1e-4)

    # Compute the dot product between true and predicted quaternions
    dot_product = tf.reduce_sum(tf.multiply(y_true_norm, y_pred_norm))
    dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)

    # Compute the quaternion distance
    distance = 2.0 * tf.acos(tf.abs(dot_product))
    distance = 180 * distance / np.pi
    return distance


matrix_size = 257
center_size = np.floor(float(matrix_size) / 2.0)

matrix = np.zeros((matrix_size, matrix_size, matrix_size))
center = np.array([center_size, center_size, center_size])
pts = []

step = (matrix_size - center_size) - 1
line_step = 0.1

for i in tqdm(np.arange(-1, 1.01, line_step)):
    for j in np.arange(-1, 1.01, line_step):
        for k in np.arange(-1, 1.01, line_step):
            end = np.array([i*step, j*step, k*step])
            pts.append((center, center+end))

reference_pt = np.array([center_size+step, center_size+step, center[0] + step])

length = center_size

ref_quat = get_quaternion(reference_pt, center)

value = 127
for start, end in tqdm(pts):
    test_quat = get_quaternion(end, center)
    value = angle_loss(ref_quat.astype(np.float32),
                       test_quat.astype(np.float32))

    if np.sum(abs(end - reference_pt)) == 0.0:
        value = 255

    draw_line_in_matrix(matrix, start, end, length, int(
        np.linalg.norm(start-end)*2), val=value)

start = center
end = reference_pt
draw_line_in_matrix(matrix, start, end, length, int(
    np.linalg.norm(start-end)*2), val=255)


thresholds = [10, 20, 30, 40]

for th in thresholds:
    mat = np.where(np.logical_or(matrix <= th, matrix == 255), matrix, 0)
    mat = np.where(mat > 0, 255, 0)
    io.imsave(f"out_sub{th}.tif", mat)

io.imsave("out.tif", matrix)
