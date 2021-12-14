# %%
import sys
import numpy as np
import IPython
from matplotlib import pyplot as plt

z = np.load(sys.argv[1])
probs,priors,like,grid,*_ = [z[k] for k in z.files]

def plot(log_probs):
  plt.contour(grid[:, :, 0], grid[:, :, 1], log_probs, zorder=1)
  plt.contourf(grid[:, :, 0], grid[:, :, 1], log_probs, zorder=0, alpha=0.55)
  plt.plot([0., 1., 0.5], [0., 0., 1.], "ro", ms=20, markeredgecolor="k")
  plt.colorbar()

# %%
IPython.embed()