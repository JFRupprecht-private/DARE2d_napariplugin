import click
import matplotlib.pyplot as plt
import json
import os
import numpy as np


def display_histogram(ax, d, mode, allowed_metrics):
    """ Display an histogram from the given dict and metric key
    d = {
        "set_1": {"loss":..., "metric":...},
        "set_2": {"loss":..., "metric":...}
    }

    Args:
        d (_type_): _description_
        key (_type_): _description_
    """
    sets = list(d.keys())
    if len(sets) == 0:
        return

    ax.set_ylabel('Metric value')
    ax.set_title(f'Metric value by sets in mode {mode}')
    ax.set_ylim(0, 1.0)

    metrics_name = d[sets[0]].keys()
    metrics_name = [
        metric for metric in metrics_name if metric in allowed_metrics]
    X = np.arange(len(metrics_name))

    item_width = 0.9 / len(sets)
    for i, _set in enumerate(sets):
        metrics = [value for k, value in d[_set].items()
                   if k in allowed_metrics]
        ax.bar(X + (item_width*i) - item_width,
               metrics, label=_set, width=item_width)

    X_axis = np.arange(len(metrics_name))
    X = metrics_name
    ax.set_xticks(X_axis, X)


def display_limit_curve_by_metric(ax, _set, metric):
    all_metric = _set["all"][metric]
    solo_metric = _set["solo"][metric]
    limits = [(float(limit_key), limit_value[metric])
              for limit_key, limit_value in _set["limit"].items()]
    limits_x, limits_y = zip(*limits)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Limit curve with metric: {metric}")
    ax.plot(limits_x, limits_y, label="limit")
    ax.plot([0.0, limits_x[-1]], [all_metric, all_metric],
            marker='o', color='red', linestyle="--", label="pretrained")
    ax.plot([0.0, limits_x[-1]], [solo_metric, solo_metric],
            marker='x', color='green', linestyle="--", label="solo")


def display_limit_curve(_set, _set_key, folder, allowed_metrics):
    metrics = _set["all"]
    metrics = [metric for metric in metrics if metric in allowed_metrics]

    fig, axs = plt.subplots(len(metrics))
    fig.suptitle(f'Limit curve with set: {_set_key}')
    for i, metric in enumerate(metrics):
        display_limit_curve_by_metric(axs[i], _set, metric)

    # box = fig.get_position()
    # fig.set_position([box.x0, box.y0 + box.height * 0.1,
    #                   box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.legend(["limit", "pretrained", "solo"], loc='upper center', bbox_to_anchor=(0.5, -0.4),
               fancybox=True, shadow=True, ncol=5)

    fig.tight_layout()
    plt.savefig(os.path.join(folder, f"limit_curve_{_set_key}.png"), dpi=900)
    plt.clf()


def display_results(scores_path, folder, allowed_metrics=["precision", "recall", "fmeasure", "loss"]):
    """Display the scores results

    scores = {
        "set_1":
            {
                "all": {"loss":..., "metric":...}  # training on all but set 1
                "solo": {"loss":..., "metric":...} # training on only set 1
                "limit":                           # training on set 1 with "all" model
                    {
                        '0.25': {"loss":..., "metric":...}
                        '0.5': {"loss":..., "metric":...}
                    }
            },
        "set_2":
    }

    Args:
        scores (_type_): _description_
    """
    if not os.path.exists(scores_path):
        raise ValueError(f"Could not find scores file at {scores_path}")
    with open(scores_path) as scores_file:
        scores = json.loads(scores_file.read())

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, axs = plt.subplots(2)
    fig.suptitle('Cross validation results')

    # Display all results (histogram)
    # Gather all "all"
    alls = {key: value["all"]
            for key, value in scores.items()}
    display_histogram(axs[0], alls, "alls", allowed_metrics)

    # Display solo results (histogram)
    solos = {key: value["solo"]
             for key, value in scores.items()}
    display_histogram(axs[1], solos, "solos", allowed_metrics)

    plt.legend(alls.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True, shadow=True, ncol=5)
    fig.tight_layout()
    plt.savefig(os.path.join(folder, f"histogram.png"), dpi=900)
    plt.clf()

    # Display limit results per set (curve) + add points "all" & "solo" as 1.0
    for _set in scores.keys():
        display_limit_curve(scores[_set], _set, folder, allowed_metrics)


@ click.command()
@ click.option(
    "--scores",
    required=True,
    help="Path to the json scores file.",


)
@ click.option(
    "--output",
    required=False,
    default="logs/output_scores",
    help="Path where to write the displayed scores."
)
def main(scores: str, output: str):
    display_results(scores, output)


if __name__ == "__main__":
    main()
