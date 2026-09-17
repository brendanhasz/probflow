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
    n_min = df["n_datapoints"].min()
    d_max = df["n_dimensions"].max()
    sub = df[
        (df["operation"] == "train")
        & (df["n_datapoints"] == n_min)
        & (df["n_dimensions"] == d_max)
    ]

    backends = sorted(sub["backend"].unique())
    eager_values = [True, False]
    width = 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, backend in enumerate(backends):
        heights = []
        for eager in eager_values:
            row = sub[(sub["backend"] == backend) & (sub["eager"] == eager)]
            heights.append(
                row["runtime_seconds"].mean() if len(row) else 0.0
            )
        xs = [j + i * width for j in range(len(eager_values))]
        ax.bar(xs, heights, width=width, label=backend, color=BACKEND_COLORS.get(backend))

    ax.set_xticks([j + width * (len(backends) - 1) / 2 for j in range(len(eager_values))])
    ax.set_xticklabels(["Eager", "Non-eager"])
    ax.set_ylabel("Training runtime (s)")
    ax.set_title(f"Eager vs non-eager training (n={n_min}, d={d_max})")
    ax.legend(title="Backend")
    fig.tight_layout()

    filename = "eager_vs_noneager.png"
    fig.savefig(os.path.join(IMG_DIR, filename))
    plt.close(fig)
    return filename


def save_backend_comparison_plots(df: pd.DataFrame) -> list[dict]:
    """Runtime vs n_datapoints at the largest dimension, lines per backend."""
    d_max = df["n_dimensions"].max()
    plots = []
    for operation in ["train", "predict", "sample"]:
        sub = df[
            (df["operation"] == operation)
            & (df["n_dimensions"] == d_max)
            & ((df["operation"] != "train") | (df["eager"] == False))  # noqa: E712
        ]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        for backend in sorted(sub["backend"].unique()):
            data = sub[sub["backend"] == backend].sort_values("n_datapoints")
            ax.plot(
                data["n_datapoints"],
                data["runtime_seconds"],
                marker="o",
                label=backend,
                color=BACKEND_COLORS.get(backend),
            )
        ax.set_xlabel("Number of datapoints")
        ax.set_ylabel("Runtime (s)")
        ax.set_xscale("log")
        ax.set_title(f"{operation.capitalize()} runtime by backend (d={d_max})")
        ax.legend(title="Backend")
        fig.tight_layout()

        filename = f"backend_comparison_{operation}.png"
        fig.savefig(os.path.join(IMG_DIR, filename))
        plt.close(fig)
        plots.append({"filename": filename, "operation": operation})
    return plots


def save_dimensionality_comparison_plots(df: pd.DataFrame) -> list[dict]:
    """Runtime vs n_datapoints, lines per dimensionality, one plot per backend/operation/eager combo."""
    plots = []
    for backend in sorted(df["backend"].unique()):
        for operation in ["train", "predict", "sample"]:
            eager_states = [True, False] if operation == "train" else [None]
            for eager in eager_states:
                sub = df[
                    (df["backend"] == backend)
                    & (df["operation"] == operation)
                ]
                if eager is None:
                    label = ""
                else:
                    sub = sub[sub["eager"] == eager]
                    label = "eager" if eager else "non-eager"
                if sub.empty:
                    continue

                fig, ax = plt.subplots(figsize=(6, 4))
                for d in sorted(sub["n_dimensions"].unique()):
                    data = sub[sub["n_dimensions"] == d].sort_values("n_datapoints")
                    ax.plot(data["n_datapoints"], data["runtime_seconds"], marker="o", label=f"d={d}")
                ax.set_xlabel("Number of datapoints")
                ax.set_ylabel("Runtime (s)")
                ax.set_xscale("log")
                title = f"{operation.capitalize()} runtime by dimensionality ({backend}"
                title += f", {label})" if label else ")"
                ax.set_title(title)
                ax.legend(title="Dimensions")
                fig.tight_layout()

                suffix = f"_{label}" if label else ""
                filename = f"dim_comparison_{backend}_{operation}{suffix}.png"
                fig.savefig(os.path.join(IMG_DIR, filename))
                plt.close(fig)
                plots.append(
                    {
                        "filename": filename,
                        "backend": backend,
                        "operation": operation,
                        "label": label,
                    }
                )
    return plots


def write_rst(
    df: pd.DataFrame,
    eager_plot: str,
    backend_plots: list[dict],
    dim_plots: list[dict],
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
        "and data dimensionality.  For each combination it "
        "measures the time to train the model, the time to generate "
        "predictions, and the time to draw samples from the model's "
        "predictive distribution.  Training is additionally benchmarked in "
        "both eager and non-eager (compiled) modes, for smaller datasets "
        "(for larger datasets non-eager/compiled mode is always used)."
    )

    eager_section = (
        "Eager vs non-eager training\n"
        "----------------------------\n\n"
        "The plot below compares training runtime in eager vs non-eager "
        "(compiled) mode for each backend, using the smallest number of "
        "datapoints and the largest number of dimensions benchmarked.\n\n"
        f".. image:: ../img/benchmarking/{eager_plot}\n"
        "   :width: 70 %\n"
        "   :align: center"
    )
    sections.append(eager_section)

    backend_lines = [
        "Comparing backends",
        "-------------------",
        "",
        "The plots below show runtime as a function of the number of "
        "datapoints, at the largest number of dimensions benchmarked, with "
        "a separate line for each backend.  Only non-eager (compiled) "
        "training runs are included.\n",
    ]
    for plot in backend_plots:
        backend_lines.append(f".. image:: ../img/benchmarking/{plot['filename']}")
        backend_lines.append("   :width: 70 %")
        backend_lines.append("   :align: center")
        backend_lines.append("")
    sections.append("\n".join(backend_lines))

    dim_lines = [
        "Comparing dimensionality",
        "-------------------------",
        "",
        "The plots below show runtime as a function of the number of "
        "datapoints, with a separate line for each number of dimensions.  "
        "Separate plots are shown for each backend, operation, and (for "
        "training) eager vs non-eager execution mode.\n",
    ]
    for plot in dim_plots:
        dim_lines.append(f".. image:: ../img/benchmarking/{plot['filename']}")
        dim_lines.append("   :width: 70 %")
        dim_lines.append("   :align: center")
        dim_lines.append("")
    sections.append("\n".join(dim_lines))

    # Full table
    cols = ["operation", "n_datapoints", "n_dimensions", "eager", "backend", "runtime_seconds"]
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
    dim_plots = save_dimensionality_comparison_plots(df)
    write_rst(df, eager_plot, backend_plots, dim_plots)


if __name__ == "__main__":
    write_benchmarking_rst_file()
