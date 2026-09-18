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
    ].copy()
    sub["mode"] = sub["eager"].map({True: "Eager", False: "Non-eager"})

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sub, x="backend", y="runtime_seconds", hue="mode", ax=ax)
    ax.set_xlabel("Backend")
    ax.set_ylabel("Training runtime (s)")
    ax.set_yscale("log")
    ax.set_title(f"Eager vs non-eager training (n={n_min}, d={d_max})")
    ax.legend(title="Mode")
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
            & ((df["operation"] != "train") | (df["eager"] == False))
        ]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
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
            f"{operation.capitalize()} runtime by backend (d={d_max})"
        )
        ax.legend(title="Backend")
        fig.tight_layout()

        filename = f"backend_comparison_{operation}.png"
        fig.savefig(os.path.join(IMG_DIR, filename))
        plt.close(fig)
        plots.append({"filename": filename, "operation": operation})
    return plots


def write_rst(
    df: pd.DataFrame,
    eager_plot: str,
    backend_plots: list[dict],
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

    # Full table
    cols = [
        "operation",
        "n_datapoints",
        "n_dimensions",
        "eager",
        "backend",
        "runtime_seconds",
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
    write_rst(df, eager_plot, backend_plots)


if __name__ == "__main__":
    write_benchmarking_rst_file()
