# %% [markdown]
# # MLP robustness on MNIST
# 
# In this notebook we evaluate the samples we collected over different runs on MNIST.

# %%
import os
from jax import numpy as jnp
import numpy as onp
import jax
import tensorflow.compat.v2 as tf
import argparse
import time
import tqdm
from collections import OrderedDict

from bnn_hmc.utils import data_utils
from bnn_hmc.utils import models
from bnn_hmc.utils import metrics
from bnn_hmc.utils import losses
from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import precision_utils

from itertools import chain

from matplotlib import pyplot as plt
import matplotlib

# %%
dtype = jnp.float32
train_set, test_set, task, data_info = data_utils.make_ds_pmap_fullbatch(
    "mnist", dtype)
test_labels = test_set[1]

net_apply, net_init = models.get_model("mlp_classification", data_info)
net_apply = precision_utils.rewrite_high_precision(net_apply)
(_, predict_fn, ensemble_upd_fn, _,_) = train_utils.get_task_specific_fns(task, data_info)

# %%
def predict_hmc(ds, basepath):
    basepath = os.path.join(basepath, "model_step_{}.pt")
    ensemble_predictions = None
    num_ensembled = 0
    last_test_predictions = None
    for i in tqdm.tqdm(range(11, 100)):
        path = basepath.format(i)
        try:
            checkpoint_dict = checkpoint_utils.load_checkpoint(path)
        except:
            continue
        _, params, net_state, _, _, accepted, _, _ = (
            checkpoint_utils.parse_hmc_checkpoint_dict(checkpoint_dict))
        if accepted:
            _, test_predictions = onp.asarray(
                predict_fn(net_apply, params, net_state, ds))
            ensemble_predictions = ensemble_upd_fn(
                ensemble_predictions, num_ensembled, test_predictions)
            num_ensembled += 1
            last_test_predictions = test_predictions
    print(num_ensembled)
    return ensemble_predictions, last_test_predictions
        

# %%
alldirs = {}
prefix = "../runs/hmc/mnist/gaussian"
dirs = os.listdir(prefix)
for path in dirs:
    full_path = os.path.join(prefix, path)
    wd = float(full_path.split("_")[4])
    stepsize = float(full_path.split("_")[6])
    trajlen = float(full_path.split("_")[8])
#     name = "Gaussian wd={} T=1".format(wd, stepsize)
    alldirs[wd] = full_path
        
alldirs

# %%
sgddirs = {}
prefix = "../runs/sgd/mnist/"

dirs = os.listdir(prefix)
for path in dirs:
    if path[:3] != "sgd":
        continue
    full_path = os.path.join(prefix, path, "model_step_99.pt")
    wd = float(full_path.split("_")[13])
    name = "SGD wd= {}".format(wd)
    sgddirs[name] = full_path
        
sgddirs

# %% [markdown]
# ## Gaussian Noise

# %%
sgd_ind_accs = {key: [] for key in sgddirs}
hmc_ind_accs = {key: [] for key in alldirs}
hmc_ens_accs = {key: [] for key in alldirs}

noise_scales = onp.linspace(0, 10., 10)
for noise_scale in noise_scales:
    noisy_test_set = test_set[0] + jax.random.normal(jax.random.PRNGKey(0), test_set[0].shape) * noise_scale, test_set[1]
    
    for key, path in chain(alldirs.items()):
        hmc_predictions, predictions = predict_hmc(noisy_test_set, path)
        if hmc_predictions is None:
            continue
        hmc_ens_accs[key].append(metrics.accuracy(hmc_predictions, test_labels))
        hmc_ind_accs[key].append(metrics.accuracy(predictions, test_labels))

    for key, path in sgddirs.items():
        checkpoint_dict = checkpoint_utils.load_checkpoint(path)
        _, params, net_state, _, _ = (
            checkpoint_utils.parse_sgd_checkpoint_dict(checkpoint_dict))
        predictions = onp.asarray(predict_fn(net_apply, params, net_state, noisy_test_set)[1])
        sgd_ind_accs[key].append(metrics.accuracy(predictions, test_labels))

# %%
plt.figure(figsize=(10, 10))
for key, _ in alldirs.items():
    if hmc_ens_accs[key] == []:
        continue
    style = "-"
    plt.plot(noise_scales, hmc_ens_accs[key], style, lw=2, label=key, ms=10)
for key, _ in sgddirs.items():
    plt.plot(noise_scales, sgd_ind_accs[key], "--", lw=2, label=key, ms=10)
plt.legend(loc=(1., 0.))
plt.ylabel("Test Accuracy", fontsize=14)
plt.xlabel("Noise Scale", fontsize=14)
plt.title("MNIST Robustness", fontsize=16)

# %%

plt.figure(figsize=(10, 10))

# vals = onp.stack([onp.asarray(v) for _, v in hmc_ens_accs.items() if v != []])
# plt.plot(vals[:, 0], vals[:, -1], "bo")

for key, _ in alldirs.items():
    
    marker="o"
    if hmc_ens_accs[key] == []:
        continue
    plt.plot(hmc_ens_accs[key][0], hmc_ens_accs[key][-1], marker, label=key)
for key, _ in sgddirs.items():
    plt.plot(sgd_ind_accs[key][0], sgd_ind_accs[key][-1], "s", label=key)
    
plt.xlabel("Original Performance")
plt.ylabel("Noise scale 10")
plt.grid()
plt.legend()

# %%
cmap = matplotlib.cm.get_cmap("tab20")

# %%
wds = sorted(alldirs)
wds = [wd for wd in wds if hmc_ens_accs[wd] != []]
alphas = onp.linspace(0.3, 1., len(wds))[::-1]

plt.figure(figsize=(3, 3))
for i, wd in enumerate(wds):
    print(i)
    std = onp.sqrt(1 / wd)
    if hmc_ens_accs[wd] == []:
        continue
    plt.plot(noise_scales, hmc_ens_accs[wd], "-", lw=2, 
             label="std {:.2f}".format(std), color=cmap(i))#, alpha=alphas[i], color="g")

plt.plot(noise_scales, sgd_ind_accs["SGD wd= 100.0"], "--k", lw=2, label="SGD", ms=10)

plt.legend()
plt.grid()
plt.ylabel("Test Accuracy", fontsize=14)
plt.xlabel("Noise Scale", fontsize=14)
plt.savefig("gaussian_priors_gaussian_noise.pdf", bbox_inches="tight")

# %% [markdown]
# ## MNIST-C

# %%
from bnn_hmc.utils import data_utils
import tensorflow_datasets as tfds

all_corruptions = tfds.image_classification.mnist_corrupted._CORRUPTIONS

# %%
def load_image_dataset(
    split, batch_size, name, repeat=False, shuffle=False,
    shuffle_seed=None
):
    ds, dataset_info = tfds.load(name, split=split, as_supervised=True,
                               with_info=True)
    num_classes = dataset_info.features["label"].num_classes
    num_examples = dataset_info.splits[split].num_examples
    num_channels = dataset_info.features['image'].shape[-1]
    
    def img_to_float32(image, label):
        return tf.image.convert_image_dtype(image, tf.float32), label

    ds = ds.map(img_to_float32).cache()
    ds_stats = data_utils._ALL_IMG_DS_STATS[data_utils.ImgDatasets("mnist")]
    def img_normalize(image, label):
        """Normalize the image to zero mean and unit variance."""
        mean, std = ds_stats
        image -= tf.constant(mean, shape=[1, 1, num_channels], dtype=image.dtype)
        image /= tf.constant(std, shape=[1, 1, num_channels], dtype=image.dtype)
        return image, label

    ds = ds.map(img_normalize)
    if batch_size == -1:
        batch_size = num_examples
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(buffer_size=10 * batch_size, seed=shuffle_seed)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds), num_classes, num_examples


def get_image_dataset(name):

    test_set, n_classes, _ = load_image_dataset("test", -1, name)
    test_set = next(iter(test_set))
    
    data_info = {
        "num_classes": n_classes
    }

    return test_set, data_info


def pmap_dataset(ds, n_devices):
    return jax.pmap(lambda x: x)(data_utils.batch_split_axis(ds, n_devices))


def make_mnistc_pmap_fullbatch(corruption, dtype, n_devices=None, truncate_to=None):
    """Make train and test sets sharded over batch dim."""
    name = get_ds_name(corruption).lower()
    n_devices = n_devices or len(jax.local_devices())
    test_set, data_info = get_image_dataset(name)
    loaded = True
    test_set = pmap_dataset(test_set, n_devices)

    test_set = test_set[0].astype(dtype), test_set[1]

    return test_set, task, data_info


def get_ds_name(corruption):
    return "mnist_corrupted/{}".format(corruption)
    

dtype = jnp.float32
test_set, task, data_info = make_mnistc_pmap_fullbatch(all_corruptions[0], dtype)
test_labels = test_set[1]

net_apply, net_init = models.get_model("mlp_classification", data_info)
net_apply = precision_utils.rewrite_high_precision(net_apply)
(_, predict_fn, ensemble_upd_fn, _,_) = train_utils.get_task_specific_fns(task, data_info)

# %%
sgd_ind_accs = {key: {} for key in sgddirs}
hmc_ind_accs = {key: {} for key in alldirs}
hmc_ens_accs = {key: {} for key in alldirs}

for corruption in all_corruptions:
    noisy_test_set, _, _ = make_mnistc_pmap_fullbatch(corruption, dtype)
    
    for key, path in alldirs.items():
        hmc_predictions, predictions = predict_hmc(noisy_test_set, path)
        if hmc_predictions is None:
            continue
        hmc_ens_accs[key][corruption] = metrics.accuracy(hmc_predictions, test_labels)
        hmc_ind_accs[key][corruption] = metrics.accuracy(predictions, test_labels)

    for key, path in sgddirs.items():
        checkpoint_dict = checkpoint_utils.load_checkpoint(path)
        _, params, net_state, _, _ = (
            checkpoint_utils.parse_sgd_checkpoint_dict(checkpoint_dict))
        predictions = onp.asarray(predict_fn(net_apply, params, net_state, noisy_test_set)[1])
        sgd_ind_accs[key][corruption] = metrics.accuracy(predictions, test_labels)

# %%
plt.figure(figsize=(10, 10))
xs = onp.arange(len(all_corruptions))
for key, _ in tempdirs.items():
    if hmc_ens_accs[key] == {}:
        continue
    style = "-"
    if hmc_ens_accs[key]["identity"] > 0.95:
        plt.plot(xs, [hmc_ens_accs[key][c] for c in all_corruptions], style, lw=2, label=key, ms=10)

for key, _ in sgddirs.items():
    plt.plot(xs, [sgd_ind_accs[key][c] for c in all_corruptions], "--", label=key)
plt.xticks(xs, all_corruptions, rotation=90)

plt.legend(loc=(1., 0.))
plt.ylabel("Test Accuracy", fontsize=14)
plt.xlabel("Corruption", fontsize=14)
plt.title("MNIST Robustness", fontsize=16)

# %%
xs = onp.arange(len(all_corruptions))
plt.bar(xs, [hmc_ens_accs["Gaussian wd=100.0 T=0.0003"][c] for c in all_corruptions],
        width=0.2, label="HMC, T=3e-4"
       )
plt.bar(xs+0.2, [hmc_ens_accs["Gaussian wd=100.0 T=0.0003"][c] for c in all_corruptions],
        width=0.2, label="HMC, T=3e-4, Ind"
       )
plt.bar(xs+0.4, [sgd_ind_accs["SGD wd= 100.0"][c] for c in all_corruptions],
        width=0.2, label="SGD"
       )

plt.xticks(xs, all_corruptions, rotation=90)
plt.legend()

# %%
cmap = matplotlib.cm.get_cmap("tab20")

# %%
wds = sorted(alldirs)
wds = [wd for wd in wds if hmc_ens_accs[wd] != {}]
num_bars = len(wds) + 1
width = 0.8 / num_bars

plt.figure(figsize=(3, 7))

ys = onp.arange(len(all_corruptions))
for i, wd in enumerate(wds):
    print(i)
    std = onp.sqrt(1 / wd)
    plt.barh(ys + i * width, [hmc_ens_accs[wd][c] for c in all_corruptions],
        height=width, label="std {:.2f}".format(std), color=cmap(i)
    )
    
plt.barh(ys + (num_bars-1) * width, [sgd_ind_accs["SGD wd= 100.0"][c] for c in all_corruptions],
        height=width, label="SGD", color="k" #cmap(len(wds)+1)
    )

plt.ylim(-0.15, 16.05)

plt.yticks(ys+0.4, all_corruptions, rotation=0);
plt.savefig("gaussian_priors_mnistc.pdf", bbox_inches="tight")

# %%
all_corruptions

# %%
wds = sorted(alldirs)
alphas = onp.linspace(0.3, 1., len(wds))[::-1]

plt.figure(figsize=(3, 3))
for i, wd in enumerate(wds):
    std = onp.sqrt(1 / wd)
    if hmc_ens_accs[wd] == []:
        continue
    plt.plot(noise_scales, hmc_ens_accs[wd], "-", lw=2, 
             label="std {:.2f}".format(std), color=cmap(i))#, alpha=alphas[i], color="g")

plt.plot(noise_scales, sgd_ind_accs["SGD wd= 100.0"], "--b", lw=2, label="SGD", ms=10, color=cmap(len(wds)+1))

plt.legend()
plt.grid()
plt.ylabel("Test Accuracy", fontsize=14)
plt.xlabel("Noise Scale", fontsize=14)
plt.savefig("gaussian_priors_gaussian_noise.pdf", bbox_inches="tight")

# %%


# %%
hmc_ens_accs["Gaussian wd=100.0 T=0.0003"]

# %%
hmc_ind_accs["Gaussian wd=100.0 T=0.0003"]

# %% [markdown]
# ## Noise along PCA directions

# %%
from sklearn.decomposition import PCA

mnist_tr_np = onp.asarray(train_set[0]).reshape(-1, 784)
mnist_te_np = onp.asarray(test_set[0]).reshape(-1, 784)

# %%
pca = PCA(n_components=50)
pca.fit(mnist_tr_np)

# %%
basis = (pca.components_)

# %%
plt.imshow((mnist_tr_np[0] + 2 * onp.random.randn(50) @ basis).reshape(28, 28))

# %%
test_set[0].shape

# %%
sgd_ind_accs = {key: [] for key in sgddirs}
hmc_ind_accs = {key: [] for key in alldirs}
hmc_ens_accs = {key: [] for key in alldirs}
n_test = 10000

noise_scales = onp.linspace(0, 10., 20)
for noise_scale in noise_scales:
    noise = (onp.random.randn(n_test, 50) @ basis).reshape(test_set[0].shape)
    noisy_test_set = test_set[0] + jnp.array(noise) * noise_scale, test_set[1]
    
    for key, path in alldirs.items():
        hmc_predictions, predictions = predict_hmc(noisy_test_set, path)
        if hmc_predictions is None:
            continue
        hmc_ens_accs[key].append(metrics.accuracy(hmc_predictions, test_labels))
        hmc_ind_accs[key].append(metrics.accuracy(predictions, test_labels))

    
    for key, path in sgddirs.items():
        checkpoint_dict = checkpoint_utils.load_checkpoint(path)
        _, params, net_state, _, _ = (
            checkpoint_utils.parse_sgd_checkpoint_dict(checkpoint_dict))
        predictions = onp.asarray(predict_fn(net_apply, params, net_state, noisy_test_set)[1])
        sgd_ind_accs[key].append(metrics.accuracy(predictions, test_labels))

# %%
wds = sorted(alldirs)
wds = [wd for wd in wds if hmc_ens_accs[wd] != []]
alphas = onp.linspace(0.3, 1., len(wds))[::-1]

plt.figure(figsize=(3, 3))

wd = 10.
std = onp.sqrt(1 / wd)
plt.plot(noise_scales, hmc_ens_accs[wd], "-", lw=2, 
         label="HMC".format(std), color="g")#, alpha=alphas[i], color="g")

plt.plot(noise_scales, sgd_ind_accs["SGD wd= 100.0"], "--k", lw=2, label="SGD", ms=10)

plt.legend()
plt.grid()
plt.ylabel("Test Accuracy", fontsize=14)
plt.xlabel("Noise Scale", fontsize=14)
plt.savefig("gaussian_prior_pca_gaussian_noise.pdf", bbox_inches="tight")

# %%


# %%

plt.figure(figsize=(10, 10))

# vals = onp.stack([onp.asarray(v) for _, v in hmc_ens_accs.items() if v != []])
# plt.plot(vals[:, 0], vals[:, -1], "bo")

for key, _ in alldirs.items():
    if hmc_ens_accs[key] == []:
        continue
    if key == "Gaussian wd=10":
        marker = "*"
    else:
        marker="o"
    plt.plot(hmc_ens_accs[key][0], hmc_ens_accs[key][-1], marker, label=key)
for key, _ in sgddirs.items():
    plt.plot(sgd_ind_accs[key][0], sgd_ind_accs[key][-1], "s", label=key)
    
plt.xlabel("Original Performance")
plt.ylabel("Noise scale 10")
plt.grid()
plt.legend()
