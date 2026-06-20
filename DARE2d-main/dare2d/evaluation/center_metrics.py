import numpy as np
import cv2
from scipy.spatial.distance import cdist
import tensorflow as tf


class CenterRecallMetric(tf.keras.metrics.Metric):
    def __init__(self, name='center_recall', max_distance=5, ** kwargs):
        super(CenterRecallMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name='tp_recall', initializer='zeros', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.false_negatives = self.add_weight(
            name='fn_recall', initializer='zeros', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.max_distance = max_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, _, fn = pred_to_stats(y_true, y_pred, self.max_distance)
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        if self.true_positives == 0:
            return self.true_positives
        return self.true_positives / (self.true_positives + self.false_negatives)

    def reset_states(self):
        """Reset the metrics state"""
        self.true_positives.assign(0)
        self.false_negatives.assign(0)


class CenterPrecisionMetric(tf.keras.metrics.Metric):
    def __init__(self, name='center_precision', max_distance=5, ** kwargs):
        super(CenterPrecisionMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name='tp_precision', initializer='zeros', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.false_positives = self.add_weight(
            name='fp_precision', initializer='zeros', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.max_distance = max_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fp, _ = pred_to_stats(y_true, y_pred, self.max_distance)
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        if self.true_positives == 0:
            return self.true_positives
        return self.true_positives / (self.true_positives + self.false_positives)

    def reset_states(self):
        """Reset the metrics state"""
        self.true_positives.assign(0)
        self.false_positives.assign(0)


class FMeasureMetric(tf.keras.metrics.Metric):
    def __init__(self, name='center_fmeasure', max_distance=5, ** kwargs):
        super(FMeasureMetric, self).__init__(name=name, **kwargs)
        self.center_precision = CenterPrecisionMetric(
            'F1_center_precision', max_distance=max_distance)
        self.center_recall = CenterRecallMetric(
            'F1_center_recall', max_distance=max_distance)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.center_precision.update_state(y_true, y_pred)
        self.center_recall.update_state(y_true, y_pred)

    def result(self):
        precision = self.center_precision.result()
        recall = self.center_recall.result()
        if precision + recall == 0:
            return precision
        return 2*precision*recall / (precision+recall)

    def reset_states(self):
        """Reset the metrics state"""
        self.center_precision.reset_states()
        self.center_recall.reset_states()


def get_centroids_distance(true_centroids, pred_centroids):
    # M = len(true), N = len(pred)
    # Dist mat = MxN
    dist_mat = cdist(true_centroids, pred_centroids, "euclidean")

    return dist_mat


def distance_wrapper(max_distance=0):
    def distance(y_true, y_pred):
        true_centroids = extract_centers(y_true)
        pred_centroids = extract_centers(y_pred)

        if len(true_centroids) > 0 and len(pred_centroids) > 0:
            dist_mat = get_centroids_distance(true_centroids, pred_centroids)
            matched_items = iterative_matching(dist_mat, max_distance)

            distances = [dist_mat[i, j] for i, j in matched_items]
            if len(distances) > 0:
                return np.mean(distances)
        return None

    return distance


def iterative_matching(dist_mat, max_distance):
    matched_items = []

    cdist_mat = dist_mat.copy()
    max_value = np.max(cdist_mat) + 1

    while(np.min(cdist_mat) < max_value):
        # Find the min value
        i, j = np.unravel_index(cdist_mat.argmin(), cdist_mat.shape)

        # If the minimum distance is above the threshold it means
        # that we won't have anymore matches so we stop
        if cdist_mat[i, j] >= max_distance:
            break

        matched_items.append((i, j))

        # "Disable" the true and pred items by setting their distance to
        # the max value + 1
        # This is a trick to avoid counting true and pred elements twice
        cdist_mat[i, :] = max_value
        cdist_mat[:, j] = max_value

    return matched_items

# Precision


def get_tp_tn_fp_fn(dist_mat, max_distance):
    # Iteratively assign centers based on distance
    tp, fp, fn = 0, 0, 0

    # FN is undetected centers
    # FP is overdetected centers
    # TP is correct centers

    matched_items = iterative_matching(dist_mat, max_distance)

    tp = len(matched_items)
    fp = dist_mat.shape[1] - tp
    fn = dist_mat.shape[0] - tp
    return tp, fp, fn


def pred_to_stats(y_true, y_pred, max_distance):
    true_centroids = extract_centers(y_true)
    pred_centroids = extract_centers(y_pred)
    tp, fp, fn = 0, 0, 0
    if len(pred_centroids) > 0 and len(true_centroids) > 0:
        dist_mat = get_centroids_distance(true_centroids, pred_centroids)
        tp, fp, fn = get_tp_tn_fp_fn(dist_mat, max_distance)
    else:
        fn = len(true_centroids)
        fp = len(pred_centroids)
    return tp, fp, fn


def precision_wrapper(max_distance=5):
    def precision(y_true, y_pred):
        tp, fp, _ = pred_to_stats(y_true, y_pred, max_distance)
        if (tp+fp) > 0:
            return tp / (tp + fp)
        return 1.0

    return precision

# Recall


def recall_wrapper(max_distance=5):
    def recall(y_true, y_pred):
        tp, _, fn = pred_to_stats(y_true, y_pred, max_distance)
        if (tp + fn) > 0:
            return tp / (tp + fn)
        return 1.0
    return recall


def extract_centers(mask):
    centroids = []
    mask = np.where(mask < 0.5, 0, 1)
    mask = mask.astype(np.uint8)

    # Extract connected components contour
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Compute the centroid of the connected components
    for cont in contours:
        M = cv2.moments(cont)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            point = (cx, cy)
            if point not in centroids:
                centroids.append(point)

    return centroids
