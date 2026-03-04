# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels.graphics.gofplots
import xarray as xr

# %%
irf = xr.load_dataset('../output/irf_1GtCO2_new_scenarios.nc')

# %%
# pooling 10 new scenarios
irf_2100 = irf['temperature'].sel(timebounds=2101).mean(dim='scenario')

# %%
plt.hist(irf_2100)

# %%
proj = xr.load_dataset('../output/scenario_projections_fair.nc')

# %%
# pooling the two main scenarios
proj.scenario

# %%
proj_2100 = proj['temperature'].sel(scenario=['2021-11-09_1__high__C__SSP1BL__exclude', '2021-11-09_1__low__U__SSP1BL__exclude'], timebounds=2101).mean(dim='scenario')

# %%
plt.hist(proj_2100)

# %%
plt.scatter(irf_2100, proj_2100)
plt.xlabel('IRF 2100 K/GtCO2')
plt.ylabel('Temperature anomaly 2100 Current policies K')

# %%
df_out = pd.DataFrame(np.array((irf_2100.data, proj_2100.data)).T, columns=['irf', 'warming'], index=irf_2100.config)

# %%
df_out.to_csv('../output/irf_warming.csv')

# %%
statsmodels.graphics.gofplots.qqplot_2samples(irf_2100.data, proj_2100.data, xlabel=None, ylabel=None, line="r", ax=None)

# %%
scipy.stats.linregress(irf_2100.data, proj_2100.data)

# %%
