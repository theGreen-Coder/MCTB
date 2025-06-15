import json
import pandas as pd
import matplotlib.pyplot as plt
import joypy
import numpy as np

def plot_distribution_results(data_file, save_file=None, benchmark_line=None, dark_mode=True):
    # Load & flatten JSON → DataFrame
    try:
        with open(data_file) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print("Invalid JSON syntax:", e)
    except FileNotFoundError:
        raise FileNotFoundError("File was not found!")
    except:
        raise "Error reading/finding the JSON file"
    
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })

    if dark_mode:
        # DARK mode styling
        plt.rcParams['font.family']       = 'Arial'
        plt.rcParams['figure.facecolor']  = 'black'
        plt.rcParams['axes.facecolor']    = 'black'
        plt.rcParams['savefig.facecolor'] = 'black'
        plt.rcParams['text.color']        = 'white'
        plt.rcParams['xtick.color']       = 'white'
        plt.rcParams['ytick.color']       = 'white'
        white_color = 'white'
    else:
        # LIGHT mode styling
        plt.rcParams['font.family']       = 'Arial'
        plt.rcParams['figure.facecolor']  = 'white'
        plt.rcParams['axes.facecolor']    = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['text.color']        = 'black'
        plt.rcParams['xtick.color']       = 'black'
        plt.rcParams['ytick.color']       = 'black'
        white_color = 'black'

    df = pd.DataFrame(
        [
            {"model": model, "score": score}
            for model, metrics in raw.items()
            for metric, scores in metrics.items()
            for score in scores
        ]
    )

    # Order models by ascending mean score, then flip for top→bottom
    order = (
        df.groupby('model')['score']
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    df['model'] = pd.Categorical(df['model'], categories=order, ordered=True)
    df = df.sort_values('model', ascending=False)

    # Plot with joypy
    fig, axes = joypy.joyplot(
        data=df,
        by='model',
        column='score',
        figsize=(6, len(order) * 1.2),
        overlap=1,
        linewidth=1,
        colormap=plt.cm.Set2,
        legend=False
    )

    # 4) Add vertical reference lines
    mean_map  = df.groupby('model')['score'].mean()
    benchmark = benchmark_line

    for ax, mdl in zip(axes, order):
        if benchmark_line and type(benchmark_line) == int:
            ax.axvline(benchmark,
                    ls='--', lw=1.2,
                    color='lime',
                    dashes=(4,3),
                    zorder=100)
        ax.axvline(mean_map[mdl],
                ls='--', lw=0.8,
                color=white_color,
                zorder=100)

    # 5) Final touches
    axes[-1].set_xlabel('Creativity score', color=white_color)
    plt.xlim(df['score'].min() - 1, df['score'].max() + 4)
    plt.tight_layout()

    if save_file:
        # Save file with a high DPI
        plt.savefig(save_file, dpi=300)

    plt.show()
