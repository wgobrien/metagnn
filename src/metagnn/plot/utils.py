import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from metagnn.utils import is_notebook

def activate_plot_settings():
    if is_notebook():
        # Set Matplotlib font types for vector graphics
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["figure.figsize"] = [6, 4]
        mpl.rcParams["savefig.dpi"] = 200
    else:
        mpl.rcParams["savefig.dpi"] = 400
        
    sns.set_theme(style="darkgrid", rc={"grid.color": ".6", "grid.linestyle": ":"})
    sns.set_context("paper", font_scale=1.)
    
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["figure.autolayout"] = False
    mpl.rcParams["legend.frameon"] = True

def plot_loss(losses: list, label: str, log_loss: bool=False):
    activate_plot_settings()
    fig,ax = plt.subplots(1, 2, figsize=(12,4), sharey=False)
    if log_loss:
        sns.lineplot(np.log10(losses), ax=ax[0])
    else:
        sns.lineplot(losses, ax=ax[0])
    ax[0].set_title(f"{label} loss")
    ax[0].set_xlabel("steps")

    sns.lineplot(np.diff(losses), ax=ax[1])
    ax[1].set_title("step delta")
    ax[1].set_xlabel("steps")

    plt.show()