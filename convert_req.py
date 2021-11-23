# %%
req_filename = 'requirements.txt'
with open(req_filename) as f:
    reqs = [l for l in f.readlines()
            if l.strip() and l.strip()[0] != '#']

# %%
pip_reqs = [r for r in reqs if r.strip().endswith('pypi_0')]
conda_reqs = [r for r in reqs if r not in pip_reqs]
pip_reqs = [r.replace('=pypi_0', '').replace('=', '==') for r in pip_reqs]

# %%
with open('pip_reqs.txt', 'w') as f:
    f.writelines(pip_reqs)
    
with open('conda_reqs.txt', 'w') as f:
    f.writelines(conda_reqs)