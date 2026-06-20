import cv2
import numpy as np
from tqdm import tqdm
from skimage import io

from dare2d.evaluation.center_metrics import (
    extract_centers,
    get_centroids_distance,
    iterative_matching,
)
from dare2d.prediction.center2d_strategy import Center2dInferenceStrategy


def display_matched_centers(
    test_set, predictions, true_matches, pred_matches, display_matched=True
):
    t_size = len(true_matches)

    ## COLORS in RGB

    # true color
    green = (0, 255, 0)
    # pred color
    blue = (0, 0, 255)
    # match but in other time frame - orange
    true_match_other = (127, 0, 255)
    # match but in other time frame - purple-ish
    pred_match_other = (200, 160, 200)
    # No match - red
    missed_true_color = (255, 0, 0)
    # No match - yellow
    missed_pred_color = (255, 255, 0)

    # Green link = match
    match_color = (0, 255, 0)

    # Has multiple matches
    multi_match_pred = (0, 127, 127)
    multi_match_true = (127, 127, 0)
    pt_size = 10

    result_array = []
    mask_array = []

    for t in range(t_size):
        has_error = False
        x = test_set.images[t]
        true_match = true_matches[t]
        pred_match = pred_matches[t]
        mask = np.zeros_like(x)

        for true_center, match in true_match.items():
            if len(match) == 0:
                cv2.line(mask, inv(true_center), inv(true_center), missed_true_color, pt_size)
                has_error = True
            if len(match) == 1 and display_matched:
                matched_item = match[0]
                if matched_item["t"] == t:
                    cv2.line(mask, inv(true_center), inv(true_center), green, pt_size)
                    cv2.line(
                        mask,
                        inv(matched_item["position"]),
                        inv(matched_item["position"]),
                        blue,
                        pt_size,
                    )
                    cv2.line(
                        mask,
                        inv(true_center),
                        inv(matched_item["position"]),
                        match_color,
                        2,
                    )
                else:
                    cv2.line(
                        mask,
                        inv(true_center),
                        inv(true_center),
                        true_match_other,
                        pt_size,
                    )
            if len(match) > 1 and display_matched:
                cv2.line(mask, inv(true_center), inv(true_center), multi_match_true, pt_size)
                for matched_item in match:
                    if matched_item["t"] == t:
                        cv2.line(
                            mask,
                            inv(matched_item["position"]),
                            inv(matched_item["position"]),
                            blue,
                            pt_size,
                        )
                        cv2.line(
                            mask,
                            inv(true_center),
                            inv(matched_item["position"]),
                            match_color,
                            pt_size,
                        )
        for (
            pred_center,
            match,
        ) in pred_match.items():
            if len(match) == 0:
                cv2.line(mask, inv(pred_center), inv(pred_center), missed_pred_color, pt_size)
                has_error = True
            if len(match) == 1 and display_matched:
                matched_item = match[0]
                if matched_item["t"] != t:
                    cv2.line(
                        mask,
                        inv(pred_center),
                        inv(pred_center),
                        pred_match_other,
                        pt_size,
                    )
                else:
                    cv2.line(mask, inv(pred_center), inv(pred_center), blue, pt_size)
            if len(match) > 1 and display_matched:
                cv2.line(mask, inv(pred_center), inv(pred_center), multi_match_pred, pt_size)

        x_display = x[:, :, -1] * 255
        x_display = cv2.cvtColor(x_display, cv2.COLOR_GRAY2RGB)
        # non_zero_mask = mask > 0
        # x_display[non_zero_mask] = mask[non_zero_mask]
        mask_array.append(mask[:, :, -1].astype(np.uint8))
        result_array.append(x_display.astype(np.uint8))

        # if has_error:
        #     io.imsave(f"logs/error_{t}.tif", mask.astype(np.uint8))
        #     io.imsave(f"logs/x_{t}.tif", x_display.astype(np.uint8))
        #     io.imsave(f"logs/pred_{t}.tif", predictions[t].astype(np.uint8))

    io.imsave(f"logs/mask.tif", np.asarray(mask_array))
    io.imsave(f"logs/predictions.tif", np.asarray(predictions))
    io.imsave(f"logs/eval.tif", np.asarray(result_array))


def get_num_matched_other_t(matching):
    num = 0
    for t in matching.keys():
        for matched in matching[t].values():
            if len(matched) > 0:
                at_least_one_at_t = False
                for match in matched:
                    if match["t"] == t:
                        at_least_one_at_t = True
                        break
                if not at_least_one_at_t:
                    num += 1
    return num


def get_num_matched(matching):
    return sum(
        [
            1 if len(matched) > 0 else 0
            for center in matching.values()
            for matched in center.values()
        ]
    )


def get_num_total(matching):
    return sum([len(centers) for centers in matching.values()])


def display_segmentation(y_true, y_pred, x):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8))
    # binary_pred = np.where(y_pred > .5, 1.0, 0)
    ax1.imshow(y_true, cmap="gray")
    ax2.imshow(y_pred, cmap="gray")
    ax3.imshow(x, cmap="gray")
    plt.pause(0)


def to_tuple(center):
    return (int(center[0]), int(center[1]))


def match_centers_at_t(
    true_centers,
    pred_centers,
    true_matching,
    pred_matching,
    true_t,
    pred_t,
    xy_threshold,
):
    for center in true_centers:
        center = to_tuple(center)
        if center not in true_matching[true_t]:
            true_matching[true_t][center] = []

    for pcenter in pred_centers:
        pcenter = to_tuple(pcenter)
        if pcenter not in pred_matching[pred_t]:
            pred_matching[pred_t][pcenter] = []

    assert len(true_matching[true_t]) == len(
        true_centers
    ), f"Expected groundtruth centers to have the same size as the matching array but found {len(true_centers)} and {len(true_matching[true_t])}"
    assert len(pred_matching[pred_t]) == len(
        pred_centers
    ), f"Expected prediction centers to have the same size as the matching array but found {len(pred_centers)} and {len(pred_matching[pred_t])}"

    if len(true_centers) > 0 and len(pred_centers) > 0:
        current_dist_mat = get_centroids_distance(true_centers, pred_centers)
        current_matched_centers = iterative_matching(current_dist_mat, max_distance=xy_threshold)

        for j, k in current_matched_centers:
            center = to_tuple(true_centers[j])
            pred_center = to_tuple(pred_centers[k])

            true_match = {
                "t": pred_t,
                "position": pred_center,
                "distance": current_dist_mat[j, k],
            }
            pred_match = {"t": true_t, "position": center}

            assert center in true_matching[true_t]
            true_matching[true_t][center].append(true_match)
            assert pred_center in pred_matching[pred_t]
            pred_matching[pred_t][pred_center].append(pred_match)

    return true_matching, pred_matching


def match_centers(true_centers, pred_centers, xy_threshold):
    # Match predicted centers with current true_centers
    # Match predicted centers with next true_centers

    # The idea is that the prediction can be "early"
    # Which leads to a false and missed detection

    t_size = len(true_centers)
    true_centers_matching = {i: {} for i in range(t_size)}
    pred_centers_matching = {i: {} for i in range(t_size)}

    for t in tqdm(range(t_size)):
        # Match in current t
        true_centers_matching, pred_centers_matching = match_centers_at_t(
            true_centers[t],
            pred_centers[t],
            true_centers_matching,
            pred_centers_matching,
            t,
            t,
            xy_threshold,
        )

        # Match in next Z
        if t < t_size - 1:
            true_centers_matching, pred_centers_matching = match_centers_at_t(
                true_centers[t + 1],
                pred_centers[t],
                true_centers_matching,
                pred_centers_matching,
                t + 1,
                t,
                xy_threshold,
            )
        if t > 0:
            true_centers_matching, pred_centers_matching = match_centers_at_t(
                true_centers[t - 1],
                pred_centers[t],
                true_centers_matching,
                pred_centers_matching,
                t - 1,
                t,
                xy_threshold,
            )
    return true_centers_matching, pred_centers_matching


def inv(pt):
    return (pt[1], pt[0])


def evaluate_center_detection(model, generator, max_distance=10, display=False):
    all_groundtruth_centers = []
    all_predicted_centers = []

    detection2d_strategy = Center2dInferenceStrategy(window_size=256, overlap=0.5)

    predictions = []

    for i in tqdm(range(len(generator))):
        # X = full image
        x = generator.images[i]
        bipoints = generator.bipoints[i]

        # Predict the centers
        prediction = detection2d_strategy.inference(x, model)
        predicted_centers = extract_centers(prediction)
        predicted_centers = [(x, y) for y, x in predicted_centers]  # flip

        predictions.append(prediction[..., -1] * 255)
        all_predicted_centers.append(predicted_centers)

        # if display:
        #     y = generator.mask_all[i]
        #     display_segmentation(y, prediction, x[:, :, 1])

        # Retrieve the ground truth centers with angle and length from the bipoints
        centers = []
        for bipoint in bipoints:
            center = (bipoint[0] + bipoint[1]) / 2
            centers.append(center)

        all_groundtruth_centers.append(centers)

    # Match all centers together
    true_centers_matching, pred_centers_matching = match_centers(
        all_groundtruth_centers, all_predicted_centers, xy_threshold=max_distance
    )

    if display:
        display_matched_centers(
            generator, predictions, true_centers_matching, pred_centers_matching
        )

    num_true_center = get_num_total(true_centers_matching)
    num_true_center_matched = get_num_matched(true_centers_matching)
    num_matched_other_t = get_num_matched_other_t(true_centers_matching)

    num_pred_center = get_num_total(pred_centers_matching)
    num_pred_center_matched = get_num_matched(pred_centers_matching)
    num_pred_matched_other_t = get_num_matched_other_t(pred_centers_matching)

    recall = num_true_center_matched / (num_true_center + 1e-6)
    precision = num_pred_center_matched / (num_pred_center + 1e-6)
    fmeasure = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        "true_center": num_true_center,
        "true_center_matched": num_true_center_matched,
        "true_matched_other_time": num_matched_other_t,
        "pred_center": num_pred_center,
        "pred_center_matched": num_pred_center_matched,
        "pred_matched_other_time": num_pred_matched_other_t,
        "recall": recall,
        "precision": precision,
        "fmeasure": fmeasure,
    }, (true_centers_matching, pred_centers_matching)
