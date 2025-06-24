import json
import pandas as pd
import matplotlib.pyplot as plt
import joypy
import numpy as np

def plot_correlation(data_file,
                     key,
                     second_key=None,
                     embedding_model=['GloVe'],
                     dark_mode=False,
                     plot_title="Correlation with Creativity Score",
                     x_axis_title="Creativity Score",
                     y_axis_title="Temperature",
                     save_file=False, 
                     file_name="plot.png"):

    # 1) Load & flatten JSON into a pandas DataFrame
    try:
        with open(data_file, "r") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in {data_file}: {e}") from None
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {data_file} was not found!") from None
    
    records = []
    for cfg_key, cfg in raw.items():
        if second_key:
            config_variable = cfg['config'][key][second_key]
        else:
            config_variable = cfg['config'][key]
        
        for model_name, value in cfg.items():
            if model_name == 'config':
                continue
            for s in value:
                records.append((config_variable, model_name, s))

    df = pd.DataFrame(records, columns=['config_variable', 'embedding_model', 'score'])
    sub = df[df['embedding_model'].isin(embedding_model)]
    
    # 2) Matplotlib style
    base_style = {
        "text.usetex": False,
        "mathtext.fontset": "cm",
    }
    theme_style = {
        "font.family": "Arial",
        "figure.facecolor": "black" if dark_mode else "white",
        "axes.facecolor": "black" if dark_mode else "white",
        "savefig.facecolor": "black" if dark_mode else "white",
        "text.color": "white" if dark_mode else "black",
        "xtick.color": "white" if dark_mode else "black",
        "ytick.color": "white" if dark_mode else "black",
    }
    plt.rcParams.update({**base_style, **theme_style})

    # 3) Linear regression plot
    x = sub['score'].values
    y = sub['config_variable'].values

    # Linear regression (fit y as a function of x)
    m, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    # Plot
    plt.scatter(x, y, label=f'{embedding_model} (T vs S)')
    x_line = np.linspace(min(x) - 0.05, max(x) + 0.05, 100)
    plt.plot(x_line, m * x_line + b,
            label=f'Fit: y={m:.2f}x + {b:.2f}\n$R$={r:.2f}')

    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)

    # 4) Finish: layout, save, show
    plt.tight_layout()

    if save_file:
        plt.savefig("./results/plots/"+file_name, dpi=300)

    plt.show()


def plot_distribution_results(data_file,
                              embedding_model=['GloVe'],
                              dark_mode=False,
                              plot_title="Creativity Score Distribution by Model",
                              x_axis_title="Creativity Score",
                              show_benchmark=True,
                              ascending=True,
                              x_min=50, 
                              x_max=100, 
                              save_file=False, 
                              file_name="plot.png"):
        
    # 1) Load & flatten JSON into a pandas DataFrame
    try:
        with open(data_file, "r") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in {data_file}: {e}") from None
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {data_file} was not found!") from None
    
    records = []
    for model, cfg in raw.items():
        for emb_model, value in cfg.items():
            if emb_model == 'config':
                continue
            for s in value:
                records.append((model, emb_model, s))

    df = pd.DataFrame(records, columns=['model', 'embedding_model', 'score'])
    df = df[df['embedding_model'].isin(embedding_model)]

    # 2) Matplotlib style
    base_style = {
        "text.usetex": False,
        "mathtext.fontset": "cm",
    }
    theme_style = {
        "font.family": "Arial",
        "figure.facecolor": "black" if dark_mode else "white",
        "axes.facecolor": "black" if dark_mode else "white",
        "savefig.facecolor": "black" if dark_mode else "white",
        "text.color": "white" if dark_mode else "black",
        "xtick.color": "white" if dark_mode else "black",
        "ytick.color": "white" if dark_mode else "black",
    }
    plt.rcParams.update({**base_style, **theme_style})
    fg_color = "white" if dark_mode else "black"

    # 3) Order models by mean
    order = (
        df.groupby("model", observed=False)["score"]
        .mean()
        .sort_values(ascending=ascending)
        .index
        .tolist()
    )
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)

    # 4) Compute mean scores & benchmark
    mean_map = df.groupby("model", observed=False)["score"].mean()
    benchmark_line = mean_map.max()
    top_model = mean_map.idxmax()

    # 5) Ridge‑plot (joypy)
    fig, axes = joypy.joyplot(
        data=df,
        by="model",
        column="score",
        figsize=(10, len(order) * 1.2),
        overlap=1,
        linewidth=1,
        colormap=plt.cm.Set1,
        legend=False,
        x_range=[x_min, x_max],
    )

    fig.suptitle(
        plot_title,
        color=fg_color,
        fontsize="x-large",
        fontweight="bold",
    )

    # 6) Vertical reference lines
    for ax, mdl in zip(axes, order):
        if show_benchmark:
            ax.axvline(
                benchmark_line,
                ls="--", lw=1.2, color="lime",
                dashes=(4, 3), zorder=100,
            )
        # per‑model mean
        ax.axvline(
            mean_map[mdl],
            ls="--", lw=0.8, color=fg_color, zorder=100,
        )

    # 7) Shared x‑label & uniform ticks
    for ax in axes:
        ax.set_xlabel("")

    axes[-1].set_xlabel(
        x_axis_title,
        color=fg_color,
        fontsize="medium",
        labelpad=12,
    )

    ticks = np.linspace(x_min, x_max, 6)
    axes[-1].set_xticks(ticks)

    # 8) Finish: layout, save, show
    plt.tight_layout()

    if save_file:
        plt.savefig("./results/plots/"+file_name, dpi=300)

    plt.show()

    print(f"Benchmark (mean of top model '{top_model}'): {benchmark_line:.2f}")
