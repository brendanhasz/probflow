"""Read benchmarking CSV results and write an RST doc summarizing them."""

import glob
import os

import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme()

CSV_GLOB = "scripts/benchmarking/benchmark_linear_regression_*.csv"
IMG_DIR = "docs/img/benchmarking"
RST_PATH = "docs/user_guide/benchmarking.rst"

# Colors assigned per-backend so they're consistent across all plots
BACKEND_COLORS = {
    "tensorflow": "tab:orange",
    "pytorch": "tab:red",
    "jax": "tab:blue",
}


def load_data() -> pd.DataFrame:
    """Load and concatenate all benchmarking CSV files."""
    files = sorted(glob.glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No CSV files found matching {CSV_GLOB}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values(
        ["backend", "n_dimensions", "n_datapoints", "operation", "eager"]
    ).reset_index(drop=True)


def df_to_list_table(df: pd.DataFrame, title: str = "") -> str:
    """Render a DataFrame as an RST list-table directive."""
    lines = [f".. list-table:: {title}".rstrip(), "   :header-rows: 1", ""]
    lines.append("   * - " + "\n     - ".join(str(c) for c in df.columns))
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("   * - " + "\n     - ".join(cells))
    return "\n".join(lines)


def save_eager_vs_noneager_plot(df: pd.DataFrame) -> str:
    """Bar chart comparing eager vs non-eager training runtime by backend."""
    device = "cpu"
    n_min = df["n_datapoints"].min()
    d_max = df["n_dimensions"].max()
    sub = df[
        (df["operation"] == "train")
        & (df["n_datapoints"] == n_min)
        & (df["n_dimensions"] == d_max)
        & (df["device"] == device)
    ].copy()
    sub["mode"] = sub["eager"].map({True: "Eager", False: "Non-eager"})
    sub["memory_usage_mb"] = sub["memory_usage"] / 1_000_000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    sns.barplot(data=sub, x="backend", y="runtime_seconds", hue="mode", ax=ax1)
    ax1.set_xlabel("Backend")
    ax1.set_ylabel("Training runtime (s)")
    ax1.set_yscale("log")
    ax1.set_title(
        f"Eager vs non-eager training (n={n_min}, d={d_max}, device={device})"
    )
    ax1.legend(title="Mode")

    sns.barplot(data=sub, x="backend", y="memory_usage_mb", hue="mode", ax=ax2)
    ax2.set_xlabel("Backend")
    ax2.set_ylabel("RAM Usage (MB)")
    ax2.legend(title="Mode")
    fig.tight_layout()

    filename = "eager_vs_noneager.png"
    fig.savefig(os.path.join(IMG_DIR, filename))
    plt.close(fig)
    return filename


def save_backend_comparison_plots(df: pd.DataFrame) -> list[dict]:
    """Runtime vs n_datapoints at the largest dimension, lines per backend."""
    d_max = df["n_dimensions"].max()
    device = "cpu"
    plots = []
    for operation in ["train", "predict", "sample"]:
        sub = df[
            (df["operation"] == operation)
            & (df["n_dimensions"] == d_max)
            & (df["device"] == device)
            & ((df["operation"] != "train") | (df["eager"] == False))
        ]
        if sub.empty:
            continue

        # only training runs have memory usage data, so only add that panel there
        if operation == "train":
            sub = sub.copy()
            sub["memory_usage_mb"] = sub["memory_usage"] / 1_000_000
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax2 = None

        sns.lineplot(
            data=sub,
            x="n_datapoints",
            y="runtime_seconds",
            hue="backend",
            hue_order=sorted(sub["backend"].unique()),
            palette=BACKEND_COLORS,
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("Number of datapoints")
        ax.set_ylabel("Runtime (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{operation.capitalize()} runtime by backend (d={d_max}, device={device})"
        )
        ax.legend(title="Backend")

        if ax2 is not None:
            sns.lineplot(
                data=sub,
                x="n_datapoints",
                y="memory_usage_mb",
                hue="backend",
                hue_order=sorted(sub["backend"].unique()),
                palette=BACKEND_COLORS,
                marker="o",
                ax=ax2,
            )
            ax2.set_xlabel("Number of datapoints")
            ax2.set_ylabel("RAM Usage (MB)")
            ax2.set_xscale("log")
            ax2.legend(title="Backend")

        fig.tight_layout()

        filename = f"backend_comparison_{operation}.png"
        fig.savefig(os.path.join(IMG_DIR, filename))
        plt.close(fig)
        plots.append({"filename": filename, "operation": operation})
    return plots


def save_cpu_vs_gpu_plots(df: pd.DataFrame) -> list[dict]:
    """Runtime vs n_datapoints (log-log), comparing CPU vs GPU, per backend."""
    plots = []
    sub_all = df[(df["eager"] == False) & (df["operation"] == "train")].copy()
    sub_all["memory_usage_mb"] = sub_all["memory_usage"] / 1_000_000
    for backend in sorted(sub_all["backend"].unique()):
        sub = sub_all[sub_all["backend"] == backend]
        if sub.empty:
            continue

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        sns.lineplot(
            data=sub,
            x="n_datapoints",
            y="runtime_seconds",
            hue="device",
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("Number of datapoints")
        ax.set_ylabel("Training runtime (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{backend.capitalize()}: CPU vs GPU training runtime")
        ax.legend(title="Device")

        sns.lineplot(
            data=sub,
            x="n_datapoints",
            y="memory_usage_mb",
            hue="device",
            marker="o",
            ax=ax2,
        )
        ax2.set_xlabel("Number of datapoints")
        ax2.set_ylabel("RAM Usage (MB)")
        ax2.set_xscale("log")
        ax2.legend(title="Device")
        fig.tight_layout()

        filename = f"cpu_vs_gpu_{backend}.png"
        fig.savefig(os.path.join(IMG_DIR, filename))
        plt.close(fig)
        plots.append({"filename": filename, "backend": backend})
    return plots


def write_rst(
    df: pd.DataFrame,
    eager_plot: str,
    backend_plots: list[dict],
    cpu_gpu_plots: list[dict],
) -> None:
    """Write the benchmarking RST document."""
    sections = []

    sections.append(
        ".. _user_guide_benchmarking:\n\n"
        "Benchmarking\n"
        "============\n\n"
        ".. include:: ../macros.hrst\n\n"
        "ProbFlow's benchmarking suite fits a Bayesian linear regression model "
        "(:class:`.LinearRegression`) with each supported backend "
        "(TensorFlow, PyTorch, and JAX), for a range of dataset sizes "
        "with 100 dimensions.  For each combination it "
        "measures the time to train the model, the time to generate "
        "predictions, and the time to draw samples from the model's "
        "predictive distribution.  Training is additionally benchmarked in "
        "both eager and non-eager (compiled) modes, for smaller datasets "
        "(for larger datasets non-eager/compiled mode is always used)."
    )

    eager_section = (
        "Eager vs compiled\n"
        "--------------------------------------\n\n"
        "The plot below compares training runtime in eager vs non-eager "
        "(compiled) mode for each backend, using the smallest number of "
        "datapoints and the largest number of dimensions benchmarked.\n\n"
        f".. image:: ../img/benchmarking/{eager_plot}\n"
        "   :width: 70 %\n"
        "   :align: center"
    )
    sections.append(eager_section)

    backend_lines = [
        "Performance by backend type",
        "---------------------------",
        "",
        (
            "The plots below show runtime as a function of the number of "
            "datapoints, at the largest number of dimensions benchmarked, with "
            "a separate line for each backend.  Only non-eager (compiled) "
            "training runs are included.\n"
        ),
        ".. tabs::",
        "",
    ]
    for plot in backend_plots:
        backend_lines.append(
            f"    .. group-tab:: {plot['operation'].capitalize()}"
        )
        backend_lines.append("")
        backend_lines.append(
            f"        .. image:: ../img/benchmarking/{plot['filename']}"
        )
        backend_lines.append("           :width: 70 %")
        backend_lines.append("           :align: center")
        backend_lines.append("")
    sections.append("\n".join(backend_lines))

    cpu_gpu_lines = [
        "Performance on CPU vs GPU",
        "-------------------------",
        "",
        (
            "The plots below show training runtime as a function of the "
            "number of datapoints (both on log scales), comparing CPU and "
            "GPU execution for each backend.  Only non-eager (compiled) "
            "training runs are included.\n"
        ),
        ".. tabs::",
        "",
    ]
    for plot in cpu_gpu_plots:
        cpu_gpu_lines.append(
            f"    .. group-tab:: {plot['backend'].capitalize()}"
        )
        cpu_gpu_lines.append("")
        cpu_gpu_lines.append(
            f"        .. image:: ../img/benchmarking/{plot['filename']}"
        )
        cpu_gpu_lines.append("           :width: 70 %")
        cpu_gpu_lines.append("           :align: center")
        cpu_gpu_lines.append("")
    sections.append("\n".join(cpu_gpu_lines))

    # Full table
    cols = [
        "operation",
        "n_datapoints",
        # "n_dimensions",
        "eager",
        "backend",
        "runtime_seconds",
        "memory_usage",
    ]
    sections.append(
        "Full benchmarking results\n"
        "-------------------------\n\n"
        "The full set of benchmarking results:\n\n"
        + df_to_list_table(df.sort_values(by=cols)[cols])
    )

    with open(RST_PATH, "w") as f:
        f.write("\n\n".join(sections) + "\n")


def write_benchmarking_rst_file() -> None:
    """Generate plots and write the benchmarking RST doc from CSV results."""
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_data()
    eager_plot = save_eager_vs_noneager_plot(df)
    backend_plots = save_backend_comparison_plots(df)
    cpu_gpu_plots = save_cpu_vs_gpu_plots(df)
    write_rst(df, eager_plot, backend_plots, cpu_gpu_plots)


if __name__ == "__main__":
    write_benchmarking_rst_file()
