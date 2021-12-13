# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import Li

sns.set()
df = pd.read_csv('scores.csv', sep='\t', header=None)
df

# %%
imdb, mnist, cifar10 = (x.set_index(0) for x in
    (df.iloc[1:8], df.iloc[10:17], df.iloc[19:26]))
algs = list(df.iloc[0,1:])

# %%
fig, axs = plt.subplots(3, 3, figsize=(12, 9))

colors = sns.color_palette("Set2", 5)
markers = '*osD^'
datasets = ['mnist', 'imdb', 'cifar10']

for col in range(3):
    ds = datasets[col]
    df = eval(ds)
    scores = df.iloc[-1:-4:-1].astype(float)
    for row in range(3):
        metric = ['ECE', 'NLL', 'Accuracy'][row]
        y = scores.iloc[row]
        y.name = metric
        ax = axs[row, col]
        if row == 0:
            ax.set_title(ds.upper(), y=1.1, fontsize=15)
        if col > 0:
            ax.set_ylabel(' ')
        else:
            ax.set_ylabel(None, fontsize=15)
        ax.xaxis.set_visible(False)
        sns.scatterplot(data=y.reset_index(), x='index', y=metric, ax=ax, s=200,
            hue='index', style='index', palette=colors)
        ymin, ymax = ax.get_ylim()
        dy = (ymax - ymin) / 15
        ax.set_ylim([ymin-dy, ymax+dy])
        ax.yaxis.set_label_coords(-.24, .5)
        ax.get_legend().set_visible(False)


handles, _ = ax.get_legend_handles_labels()
fig.tight_layout()
fig.legend(handles=handles, labels=algs, loc='right', markerscale=1.5, bbox_to_anchor=(1.1, 0.5))

# %%
fig.savefig('scores.png', dpi=160, bbox_inches='tight')

# %%
