# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
fair_params_1_4_0 = '../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv'
df_configs = pd.read_csv(fair_params_1_4_0, index_col=0)
configs = df_configs.index  # label for the "config" axis

# %%
species, properties = read_properties(filename='../data/fair-calibration/species_configs_properties_1.4.0.csv')

# %%
f = FAIR(ch4_method='Thornhill2021')
f.define_time(1750, 2101, 1)
f.define_scenarios(scenarios_list)
f.define_configs(configs)
f.define_species(species, properties)
f.allocate()

# %%
f.fill_from_csv(
    forcing_file='../data/forcing/volcanic_solar.csv',
)

# %%
# populate historical with the AR6 history
f.emissions.loc[dict(timepoints=np.arange(1750.5, 2015))] = fe.drop_vars(("scenario", "config")) * np.ones((1, 10, output_ensemble_size, 1))

# %%
del variable_mapping['Emissions|Halon1202']

# %%
# populate the future. This isn't fast...
for scenario in scenarios_list:
    for variable in variable_mapping:
        f.emissions.loc[dict(timepoints=np.arange(2015.5, 2101), specie=variable_mapping[variable], scenario=scenario)] = (
            filtered_emissions_df.loc[(filtered_emissions_df['scenario']==scenario) & (filtered_emissions_df['variable']==variable), 2015.5:].values.T
        )
        #f.emissions.loc[dict(timepoints=np.arange(2015.5, 2101), specie=specie, scenario=scenario)] = 

# %%
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

# %%
f.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
f.override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")

# %%
# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)
f.run(progress=True)

# %%
# rebase to 1850-1900
weights = np.zeros((352, 10, 841))
weights[100, ...] = 0.5
weights[101:151, ...] = 1
weights[151, ...] = 0.5
weights = xr.DataArray(
    weights, 
    dims=f.temperature.sel(layer=0).dims, 
    coords=f.temperature.sel(layer=0).coords
)
    
temperature_rebased = (
    f.temperature.sel(layer=0) - f.temperature.sel(layer=0).weighted(weights).mean(dim="timebounds")
).sel(timebounds=np.arange(1850, 2102))

# %%
tabcols = list(mc.TABLEAU_COLORS.keys())

# %%
for iscen, scenario in enumerate(scenarios_list):
    plt.fill_between(
        np.arange(1850, 2102), 
        temperature_rebased.sel(scenario=scenario).quantile(.05, dim='config'),
        temperature_rebased.sel(scenario=scenario).quantile(.95, dim='config'),
        color=tabcols[iscen],
        alpha=.05
    );
    plt.plot(np.arange(1850, 2102), temperature_rebased.sel(scenario=scenario).median(dim='config'), color=tabcols[iscen]);
plt.grid()
plt.xlim(1850, 2100)

# %%
temperature_rebased

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['timebounds', 'scenario', 'config'], temperature_rebased.data),
    ),
    coords = dict(
        scenario = scenarios_list,
        timebounds = temperature_rebased.timebounds.data.astype(int),
        config = df_configs.index
    ),
    attrs = dict(units = 'K_relative_to_1850-1900')
)
ds

# %%
os.makedirs('../output/', exist_ok=True)

# %%
ds.to_netcdf('../output/scenario_projections_fair.nc')

# %%
