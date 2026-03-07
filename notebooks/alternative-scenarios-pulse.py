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
import os

import matplotlib.colors as mc
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
scenario_df = pd.read_csv('../data/emissions/scenario_labels.csv')

# %%
scenario_df['scenario']

# %%
scenarios_list = list(scenario_df['scenario'])
scenarios_list

# %%
# Note that this file is not included in the GitHub repo owing to its size.
emissions_df = pd.read_csv('../data/emissions/scenarios_12Nov2021a_CR.csv')

# %%
filtered_emissions_df = emissions_df.loc[emissions_df['scenario'].isin(scenarios_list)]
filtered_emissions_df

# %%
# make fair years
for year in range(2015, 2101):
    filtered_emissions_df = filtered_emissions_df.rename(columns={f'{year}-01-01': year+0.5})

# %%
# convert units
for variable in ['Emissions|CO2|MAGICC AFOLU', 'Emissions|CO2|MAGICC Fossil and Industrial', 'Emissions|N2O']:
    filtered_emissions_df.loc[filtered_emissions_df['variable']==variable, 2015.5:] = filtered_emissions_df.loc[filtered_emissions_df['variable']==variable, 2015.5:]/1000

# %%
filtered_emissions_df

# %%
# delete extra columns
filtered_emissions_df = filtered_emissions_df.drop(
    columns=[
        'ambition', 
        'conditionality', 
        'country_extension', 
        'exclude_hot_air', 
        'global_extension', 
        'model', 
        'model_version',
        'pathway_id',
        'stage'
    ]
)

# %%
filtered_emissions_df

# %%
# now we need to tack the historical on to the start of this
da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500.nc")
output_ensemble_size = 841

# %%
da_emissions

# %%
da = da_emissions.sel(scenario='ssp126', timepoints=np.arange(1750.5, 2015), config='unspecified')
da

# %%
fe = da.expand_dims(dim=["scenario", "config"], axis=(1, 2))
fe

# %%
magicc_variables = filtered_emissions_df.variable.unique()

# %%
magicc_variables

# %%
variable_mapping = {mv: mv[10:] for mv in magicc_variables}

# %%
for key, value in variable_mapping.items():
    if 'FC' in key:
        variable_mapping[key] = value.replace('FC', 'FC-')

# %%
for key, value in variable_mapping.items():
    if 'Halon' in key:
        variable_mapping[key] = value.replace('Halon', 'Halon-')

# %%
variable_mapping['Emissions|CO2|MAGICC AFOLU'] = 'CO2 AFOLU'
variable_mapping['Emissions|CO2|MAGICC Fossil and Industrial'] = 'CO2 FFI'

# %%
variable_mapping['Emissions|cC4F8'] = 'c-C4F8'

# %%
del variable_mapping['Emissions|Halon1202']

# %%
fair_params_1_4_0 = '../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv'
df_configs = pd.read_csv(fair_params_1_4_0, index_col=0)
configs = df_configs.index  # label for the "config" axis

# %%
species, properties = read_properties(filename='../data/fair-calibration/species_configs_properties_1.4.0.csv')

# %%
batch_size=2
nbatches = int(np.ceil(len(scenarios_list) / batch_size))
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
    batch_end = int(np.min(((ibatch+1)*batch_size, len(scenarios_list))))
    batch_size = batch_end - batch_start
    batch_scenarios = scenarios_list[batch_start:batch_end]

    f = FAIR(ch4_method='Thornhill2021')
    f.define_time(1750, 2101, 1)
    f.define_scenarios(batch_scenarios)
    f.define_configs(configs)
    f.define_species(species, properties)
    f.allocate()

    f.fill_from_csv(
        forcing_file='../data/forcing/volcanic_solar.csv',
    )

    # populate historical with the AR6 history
    f.emissions.loc[dict(timepoints=np.arange(1750.5, 2015))] = fe.drop_vars(("scenario", "config")) * np.ones((1, batch_size, output_ensemble_size, 1))

    # populate the future. This isn't fast...
    for scenario in batch_scenarios:
        for variable in variable_mapping:
            f.emissions.loc[dict(timepoints=np.arange(2015.5, 2101), specie=variable_mapping[variable], scenario=scenario)] = (
                filtered_emissions_df.loc[(filtered_emissions_df['scenario']==scenario) & (filtered_emissions_df['variable']==variable), 2015.5:].values.T
            )

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
    f_irf.define_time(1750, 2101, 1)
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
        irf[scenario] = (f_irf.temperature-f.temperature).sel(scenario=scenario, layer=0, timebounds=np.arange(year_of_pulse, 2102))

# %% [markdown]
# ### IRF is the difference of the run with an additional 1 tCO2 pulse in 2024
#
# Sense check: IRFs on page 17 of https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
#
# note this is one model with a higher ECS than the AR6 assessment, so really this is bang in line
#
# the IRFs are the differences between the runs with an additional 1 tCO2 and the base scenarios.

# %%
irf['2021-11-09_1__high__C__SSP1BL__exclude'] 

# %%
os.makedirs('../plots', exist_ok=True)
tabcols = list(mc.TABLEAU_COLORS.keys())

# %%
fig, ax = plt.subplots(2, 5, figsize=(16, 7))
for iscen, scenario in enumerate(scenarios_list):
    i = iscen//5
    j = iscen % 5
    ax[i, j].fill_between(
        np.arange(-1, 76),
        irf[scenario].min(dim='config'), 
        irf[scenario].max(dim='config'), 
        color=tabcols[iscen], 
        alpha=0.2
    );
    ax[i, j].fill_between(
        np.arange(-1, 76),
        irf[scenario].quantile(.05, dim='config'), 
        irf[scenario].quantile(.95, dim='config'), 
        color=tabcols[iscen], 
        alpha=0.2
    );
    ax[i, j].fill_between(
        np.arange(-1, 76),
        irf[scenario].quantile(.16, dim='config'), 
        irf[scenario].quantile(.84, dim='config'), 
        color=tabcols[iscen], 
        alpha=0.2
    );
    ax[i, j].plot(np.arange(-1, 76), irf[scenario].median(dim='config'), color=tabcols[iscen]);
    ax[i, j].set_xlim(0, 75)
    ax[i, j].set_ylim(-0.1e-3, 1.2e-3)
    ax[i, j].set_ylabel('Temperature increase, K')
    ax[i, j].set_title(f'1 GtCO2 upon {scenario}')

fig.tight_layout()
plt.savefig(f'../plots/new_scenarios_irf.png')

# %%
plt.fill_between(
    np.arange(-1, 76),
    (irf[scenarios_list[9]]-irf[scenarios_list[0]]).min(dim='config'), 
    (irf[scenarios_list[9]]-irf[scenarios_list[0]]).max(dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 76),
    (irf[scenarios_list[9]]-irf[scenarios_list[0]]).quantile(.05, dim='config'), 
    (irf[scenarios_list[9]]-irf[scenarios_list[0]]).quantile(.95, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 76),
    (irf[scenarios_list[9]]-irf[scenarios_list[0]]).quantile(.16, dim='config'), 
    (irf[scenarios_list[9]]-irf[scenarios_list[0]]).quantile(.84, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.plot(np.arange(-1, 76), (irf[scenarios_list[9]]-irf[scenarios_list[0]]).median(dim='config'), color='k');
plt.xlim(0, 75)
plt.ylim(-1.2e-3, 1.2e-3)
plt.ylabel('Temperature increase, K')
plt.title('Difference between two scenarios')

plt.savefig('../plots/diff_new_scenarios_irf.png')

# %%
output = np.stack(
    (
        irf['2021-11-09_1__high__C__SSP1BL__exclude'].data, 
        irf['2021-11-09_1__low__U__SSP1BL__exclude'].data, 
        irf['2021-11-09_1__high__C__2030__exclude'].data, 
        irf['2021-11-09_1__high__C__2030__include'].data, 
        irf['2021-11-09_1__low__C__2030__exclude'].data, 
        irf['2021-11-09_1__low__C__2030__include'].data, 
        irf['2021-11-09_1__high__U__2030__exclude'].data, 
        irf['2021-11-09_1__low__U__2030__exclude'].data,
        irf['2021-11-09_1__high__U__2030__include'].data, 
        irf['2021-11-09_1__low__U__2030__include'].data,
    ), axis=0
)
output.shape

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['scenario', 'timebounds', 'config'], output),
    ),
    coords = dict(
        scenario = [f'irf_{scenario}' for scenario in scenarios_list],
        timebounds = irf['2021-11-09_1__high__C__SSP1BL__exclude']["timebounds"].data.astype(int),
        config = df_configs.index
    ),
    attrs = dict(units = 'K/GtCO2')
)
ds

# %%
os.makedirs('../output/', exist_ok=True)

# %%
ds.to_netcdf('../output/irf_1GtCO2_new_scenarios.nc')

# %%
