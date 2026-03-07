# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .conda
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
irf_2100 = (
    irf["temperature"]
    .sel(
        scenario=[
            "irf_2021-11-09_1__high__C__SSP1BL__exclude",
            "irf_2021-11-09_1__low__U__SSP1BL__exclude",
        ],
        timebounds=2101,
    )
    .to_dataframe()
    .reset_index()
    .assign(scenario=lambda df: df["scenario"].str.removeprefix("irf_"))
    .rename(columns={"temperature": "irf"})
)
irf_2100

# %%
plt.hist(irf_2100["irf"])

# %%
proj = xr.load_dataset('../output/scenario_projections_fair.nc')

# %%
# pooling the two main scenarios
proj.scenario

# %%
proj_2100 = (
    proj["temperature"]
    .sel(
        scenario=[
            "2021-11-09_1__high__C__SSP1BL__exclude",
            "2021-11-09_1__low__U__SSP1BL__exclude",
        ],
        timebounds=2101,
    )
    .to_dataframe()
    .reset_index()
)

# %%
plt.hist(proj_2100["temperature"])

# %%
df = irf_2100.merge(proj_2100, on=["scenario", "config", "timebounds"])
plt.scatter(df["irf"], df["temperature"])
plt.xlabel('IRF 2100 K/GtCO2')
plt.ylabel('Temperature anomaly 2100 Current policies K')

# %%
df_out = df[["scenario", "config", "irf", "temperature"]].set_index(["scenario", "config"])
df_out

# %%
df_out.to_csv('../output/irf_warming.csv')

# %%
statsmodels.graphics.gofplots.qqplot_2samples(df["irf"], df["temperature"], xlabel=None, ylabel=None, line="r", ax=None)

# %%
scipy.stats.linregress(df["irf"], df["temperature"])

# %%
