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

# %% [markdown]
# # Scenario evolving impulse response functions with fair
#
# For numerical stability, we want to make the pulse sizes reasonable, so add 1 GtCO2 in 2024 (about 2.5% of current CO2 emissions, and a factor of about 367 smaller than Joos who used 100 GtC ≈ 367 GtCO2).

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.autonotebook import tqdm

import fair
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
fair.__version__

# %% [markdown]
# ## First, a historical all-forcings run
#
# Now we have to break this into four as the eight scenarios do not quite fit in memory

# %%
fair_params_1_4_0 = '../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv'
df_configs = pd.read_csv(fair_params_1_4_0, index_col=0)
configs = df_configs.index  # label for the "config" axis

# %%
species, properties = read_properties(filename='../data/fair-calibration/species_configs_properties_1.4.0.csv')

# %%
# I was lazy and didn't convert emissions to CSV, so use the old clunky method of importing from netCDF
# this is from calibration-1.4.0
da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500.nc")
output_ensemble_size = 841

# %%
scenarios = [
    'ssp119',
    'ssp126',
    'ssp245',
    'ssp370',
    'ssp434',
    'ssp460',
    'ssp534-over',
    'ssp585'
]
batch_size=2

# %%
nbatches = int(np.ceil(len(scenarios) / batch_size))
nbatches

# %%
year_of_pulse = 2025
simulation_start = 1750
index_of_pulse = year_of_pulse - simulation_start
index_of_pulse

# %%
irf = {}

# %%
for ibatch in tqdm(range(nbatches)):
    batch_start = ibatch*batch_size
    batch_end = int(np.min(((ibatch+1)*batch_size, len(scenarios))))
    batch_size = batch_end - batch_start
    batch_scenarios = scenarios[batch_start:batch_end]

    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, 2500, 1)
    f.define_scenarios(batch_scenarios)
    f.define_configs(configs)
    f.define_species(species, properties)
    f.allocate()

    f.fill_from_csv(
        forcing_file='../data/forcing/volcanic_solar.csv',
    )

    da = da_emissions.loc[dict(config="unspecified", scenario=batch_scenarios)]
    fe = da.expand_dims(dim=["config"], axis=(2))
    f.emissions = fe.drop_vars(("config")) * np.ones((1, 1, output_ensemble_size, 1))

    fill(
        f.forcing,
        f.forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f.forcing,
        f.forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
        specie="Solar",
    )
    
    f.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
    f.override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")
    
    # initial conditions
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    f.run(progress=False)

    new_emissions = f.emissions.copy()
    new_emissions[index_of_pulse, :, :, 0] = new_emissions[index_of_pulse, :, :, 0] + 1

    f_irf = FAIR(ch4_method='Thornhill2021')
    f_irf.define_time(1750, 2500, 1)
    f_irf.define_scenarios(batch_scenarios)
    f_irf.define_configs(configs)
    f_irf.define_species(species, properties)
    f_irf.allocate()
    f_irf.fill_from_csv(
        forcing_file='../data/forcing/volcanic_solar.csv',
    )
    f_irf.emissions = new_emissions
    fill(
        f_irf.forcing,
        f_irf.forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f_irf.forcing,
        f_irf.forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
        specie="Solar",
    )
    
    f_irf.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
    f_irf.override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")
    
    # initial conditions
    initialise(f_irf.concentration, f_irf.species_configs["baseline_concentration"])
    initialise(f_irf.forcing, 0)
    initialise(f_irf.temperature, 0)
    initialise(f_irf.cumulative_emissions, 0)
    initialise(f_irf.airborne_emissions, 0)
    
    f_irf.run(progress=False)

    for scenario in batch_scenarios:
        irf[scenario] = (f_irf.temperature-f.temperature).sel(scenario=scenario, layer=0, timebounds=np.arange(year_of_pulse, 2501))

# %% [markdown]
# ### IRF is the difference of the run with an additional 1 tCO2 pulse in 2024
#
# Sense check: IRFs on page 17 of https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
#
# note this is one model with a higher ECS than the AR6 assessment, so really this is bang in line
#
# the IRFs are the differences between the runs with an additional 1 tCO2 and the base scenarios.

# %%
irf['ssp245']

# %%
os.makedirs('../plots', exist_ok=True)
ipcc_colors = {
    'ssp119': '#00a9cf',
    'ssp126': '#003466',
    'ssp245': '#f69320',
    'ssp370': '#df0000',
    'ssp434': '#2274ae',
    'ssp460': '#b0724e',
    'ssp534-over': '#92397a',
    'ssp585': '#980002',
}	

# %%
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for iscen, scenario in enumerate(scenarios):
    i = iscen//4
    j = iscen % 4
    ax[i, j].fill_between(
        np.arange(-1, 475),
        irf[scenario].min(dim='config'), 
        irf[scenario].max(dim='config'), 
        color=ipcc_colors[scenario], 
        alpha=0.2
    );
    ax[i, j].fill_between(
        np.arange(-1, 475),
        irf[scenario].quantile(.05, dim='config'), 
        irf[scenario].quantile(.95, dim='config'), 
        color=ipcc_colors[scenario], 
        alpha=0.2
    );
    ax[i, j].fill_between(
        np.arange(-1, 475),
        irf[scenario].quantile(.16, dim='config'), 
        irf[scenario].quantile(.84, dim='config'), 
        color=ipcc_colors[scenario], 
        alpha=0.2
    );
    ax[i, j].plot(np.arange(-1, 475), irf[scenario].median(dim='config'), color=ipcc_colors[scenario]);
    ax[i, j].set_xlim(0, 475)
    ax[i, j].set_ylim(-0.1e-3, 1.2e-3)
    ax[i, j].set_ylabel('Temperature increase, K')
    ax[i, j].set_title(f'1 GtCO2 upon {scenario}')

fig.tight_layout()
plt.savefig(f'../plots/scenarios.png')

# %%
plt.fill_between(
    np.arange(-1, 475),
    (irf['ssp585']-irf['ssp245']).min(dim='config'), 
    (irf['ssp585']-irf['ssp245']).max(dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf['ssp585']-irf['ssp245']).quantile(.05, dim='config'), 
    (irf['ssp585']-irf['ssp245']).quantile(.95, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf['ssp585']-irf['ssp245']).quantile(.16, dim='config'), 
    (irf['ssp585']-irf['ssp245']).quantile(.84, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.plot(np.arange(-1, 475), (irf['ssp585']-irf['ssp245']).median(dim='config'), color='k');
plt.xlim(0, 475)
plt.ylim(-1.2e-3, 1.2e-3)
plt.ylabel('Temperature increase, K')
plt.title('Difference ssp585 to ssp245')

plt.savefig('../plots/diff_ssp585_ssp245.png')

# %%
plt.fill_between(
    np.arange(-1, 475),
    (irf['ssp245']-irf['ssp119']).min(dim='config'), 
    (irf['ssp245']-irf['ssp119']).max(dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf['ssp245']-irf['ssp119']).quantile(.05, dim='config'), 
    (irf['ssp245']-irf['ssp119']).quantile(.95, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf['ssp245']-irf['ssp119']).quantile(.16, dim='config'), 
    (irf['ssp245']-irf['ssp119']).quantile(.84, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.plot(np.arange(-1, 475), (irf['ssp245']-irf['ssp119']).median(dim='config'), color='k');
plt.xlim(0, 475)
plt.ylim(-1.2e-3, 1.2e-3)
plt.ylabel('Temperature increase, K')
plt.title('Difference ssp245 to ssp119')

plt.savefig('../plots/diff_ssp245_ssp119.png')

# %%
output = np.stack(
    (
        irf['ssp119'].data, 
        irf['ssp126'].data, 
        irf['ssp245'].data, 
        irf['ssp370'].data, 
        irf['ssp434'].data, 
        irf['ssp460'].data, 
        irf['ssp534-over'].data, 
        irf['ssp585'].data
    ), axis=0
)
output.shape

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['scenario', 'timebounds', 'config'], output),
    ),
    coords = dict(
        scenario = [f'irf_{scenario}' for scenario in scenarios],
        timebounds = irf['ssp119']["timebounds"].data.astype(int),
        config = df_configs.index
    ),
    attrs = dict(units = 'K/GtCO2')
)
ds

# %%
os.makedirs('../output/', exist_ok=True)

# %%
ds.to_netcdf('../output/irf_1GtCO2.nc')
