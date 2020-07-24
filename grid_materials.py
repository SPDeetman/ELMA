# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:53:21 2020
@author: deetman@cml.leidenuniv.nl

This code is provided for review purposes only, please do not use the code or results without consent of the author

This module is used to calculate the materials involved in the electricity grid
Input:  1) scenario files from the IMAGE Integrated Assessment Model
        2) data files on material intensities, costs, lifetimes and weights
Output: Total global material use (stocks, inflow & outflow) in the electricity sector for the period 2000-2050 based on:
        1) the second Shared socio-economic pathway (SSP2) Baseline
        2) the second Shared socio-economic pathway (SSP2) 2-degree Climate Policy scenario

"""
# define imports, counters & settings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from past.builtins import execfile

YOUR_DIR = 'C:\\Users\\...'
os.chdir(YOUR_DIR)

datapath = YOUR_DIR + '\\grid_data\\'

execfile('read_mym.py')         # module to read in mym (IMAGE output) files
idx = pd.IndexSlice             # needed for slicing multi-index dataframes (pandas)
scenario = "SSP2" 
variant  = "450"                # Select scenario here: 'BL' = SSP2 Baseline; '450' = SSP2 2-degree climate policy scenario (450 refers to the kyoto greenhouse gas concentration limit of 450 ppm CO2-eq units)
path = scenario + "\\" +  scenario + "_" + variant + "\\"

startyear = 1971
endyear = 2100
years = endyear - startyear  + 1
regions = 26
material_select = 'Steel'

#%% Read in files --------------------------------------------------------------------------

# IMAGE file: Generation capacity (stock) in MW peak capacity (Used as proxy for grid growth)
gcap_BL_data = read_mym_df('SSP2\\SSP2_BL\\Gcap.out')
gcap_BL_data = gcap_BL_data.loc[~gcap_BL_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_BL = pd.pivot_table(gcap_BL_data, index=['time','DIM_1'], values=list(range(1,29)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

gcap_data = read_mym_df(path + 'Gcap.out')
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data, index=['time','DIM_1'], values=list(range(1,29)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

# read in additional (non-IMAGE) grid data files (grid length, grid additions & material composition)
grid_length_Hv = pd.read_csv(datapath + 'grid_length_Hv.csv', index_col=0).transpose()                          # lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
ratio_Hv = pd.read_csv(datapath + 'Hv_ratio.csv', index_col=0)                                                  # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
underground_ratio = pd.read_csv(datapath + 'underground_ratio.csv', index_col=0)                                # these contain the definition of the constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
materials_grid = pd.read_csv(datapath + 'Materials_grid.csv', index_col=0).transpose()                          # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
grid_additions = pd.read_csv(datapath + 'grid_additions.csv', index_col=0)                                      # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)
materials_grid_additions = pd.read_csv(datapath + 'Materials_grid_additions.csv', index_col=0).transpose()      # Additional infrastructure required for grid connections, such as transformers & substations (material compositin in kg/unit)
lifetime_grid_elements = pd.read_csv(datapath + 'operational_lifetime.csv', index_col=0)                        # Average lifetime in years
region_list = list(grid_length_Hv.columns.values)
material_list = list(materials_grid.index.values)

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap_techlist = ['Solar PV', 'CSP', 'Wind onshore', 'Wind offshore', 'Hydro', 'Other Renewables', 'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 'CHP Biomass + CCS']
gcap.index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), region_list], names=['years', 'regions'])
gcap_BL.index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_techlist
gcap_BL.columns = gcap_techlist

# IMAGE file: GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
gdp_pc = pd.read_csv(path + 'gdp_pc.csv', index_col=0)  
gdp_pc.columns = region_list
gdp_pc = gdp_pc.drop([1970])

#%% length calculations ----------------------------------------------------------------------------

# only the regional total (peak) generation capacity is used as a proxy for the grid growth (BL to 2016, then BL or 450)
gcap_BL_total = gcap_BL.sum(axis=1).unstack()
gcap_BL_total = gcap_BL_total[region_list]               # re-order columns to the original TIMER order
gcap_growth = gcap_BL_total / gcap_BL_total.loc[2016]    # define growth according to 2016 as base year
gcap_total = gcap.sum(axis=1).unstack()
gcap_total = gcap_total[region_list]                     # re-order columns to the original TIMER order
gcap_growth.loc[2016:endyear] = gcap_total.loc[2016:endyear] / gcap_total.loc[2016]        # define growth according to 2016 as base year

# Hv length (in kms) is region-specific. However, we use a single ratio between the length of Hv and Mv networks, the same applies to Lv networks 
grid_length_Mv = grid_length_Hv.mul(ratio_Hv['Hv to Mv'])
grid_length_Lv = grid_length_Hv.mul(ratio_Hv['Hv to Lv'])

# define grid length over time (fixed in 2016, growth according to gcap)
grid_length_Hv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Mv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Lv_time = pd.DataFrame().reindex_like(gcap_total)

for year in range(startyear, endyear+1):
    grid_length_Hv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Hv.loc['2016'])
    grid_length_Mv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Mv.loc['2016'])
    grid_length_Lv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Lv.loc['2016'])

# define underground vs. aboveground fraction (%) based on static ratios (the Hv length is the aboveground fraction according to Open Street Maps, we add the underground fractions for 3 voltage networks)
function_Hv_under = gdp_pc * underground_ratio.loc['mult','Hv'] + underground_ratio.loc['add','Hv']
function_Mv_under = gdp_pc * underground_ratio.loc['mult','Mv'] + underground_ratio.loc['add','Mv']
function_Lv_under = gdp_pc * underground_ratio.loc['mult','Lv'] + underground_ratio.loc['add','Lv']

# maximize linear function at 100 & minimize at 0 (%)
function_Hv_under = function_Hv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])                     
function_Hv_under = function_Hv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
function_Mv_under = function_Mv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
function_Mv_under = function_Mv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
function_Lv_under = function_Lv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
function_Lv_under = function_Lv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])

# MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
grid_length_Hv_total = grid_length_Hv_time                                      # assuming the length from OSM IS the abovegrond fraction
grid_length_Hv_above = grid_length_Hv_total * (1 - (function_Hv_under/100)) 
grid_length_Hv_under = grid_length_Hv_total * function_Hv_under/100

grid_length_Mv_above = grid_length_Mv_time * (1 - function_Mv_under/100)
grid_length_Mv_under = grid_length_Mv_time * function_Mv_under/100
grid_length_Mv_total = grid_length_Mv_above + grid_length_Mv_under

grid_length_Lv_above = grid_length_Lv_time * (1 - function_Lv_under/100)
grid_length_Lv_under = grid_length_Lv_time * function_Lv_under/100
grid_length_Lv_total = grid_length_Lv_above + grid_length_Lv_under

grid_subst_Hv = grid_length_Hv_total.mul(grid_additions.loc['Substations','Hv'])        # number of substations on Hv network
grid_subst_Mv = grid_length_Mv_total.mul(grid_additions.loc['Substations','Mv'])        # # of substations
grid_subst_Lv = grid_length_Lv_total.mul(grid_additions.loc['Substations','Lv'])        # # of substations
grid_trans_Hv = grid_length_Hv_total.mul(grid_additions.loc['Transformers','Hv'])       # number of transformers on the Hv network
grid_trans_Mv = grid_length_Mv_total.mul(grid_additions.loc['Transformers','Mv'])       # # of transformers
grid_trans_Lv = grid_length_Lv_total.mul(grid_additions.loc['Transformers','Lv'])       # # of transformers

#%% material calculations -------------------------------------------------------------------------------------
items = ['HV overhead', 'HV underground', 'HV substations', 'HV transformers', 'MV overhead', 'MV underground', 'MV substations', 'MV transformers',  'LV overhead', 'LV underground', 'LV substations', 'LV transformers']
columns = pd.MultiIndex.from_product([region_list, items], names=['regions', 'elements'])
index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), material_list], names=['years', 'materials'])
grid_stock_materials = pd.DataFrame(index=index, columns=columns)

for material in material_list:
    grid_stock_materials.loc[idx[:,material],idx[:,'HV overhead']]      = grid_length_Hv_above.to_numpy() * materials_grid.loc[material,'HV overhead']                      # kms * kg/km = kg
    grid_stock_materials.loc[idx[:,material],idx[:,'HV underground']]   = grid_length_Hv_under.to_numpy() * materials_grid.loc[material,'HV underground']
    grid_stock_materials.loc[idx[:,material],idx[:,'HV substations']]   = grid_subst_Hv.to_numpy()        * materials_grid_additions.loc[material,'Hv Substation']          # kms * units/km * kg/unit = kg
    grid_stock_materials.loc[idx[:,material],idx[:,'HV transformers']]  = grid_trans_Hv.to_numpy()        * materials_grid_additions.loc[material,'Hv Transformer']        # kms * units/km * kg/unit = kg
    grid_stock_materials.loc[idx[:,material],idx[:,'MV overhead']]      = grid_length_Mv_above.to_numpy() * materials_grid.loc[material,'MV overhead']
    grid_stock_materials.loc[idx[:,material],idx[:,'MV underground']]   = grid_length_Mv_under.to_numpy() * materials_grid.loc[material,'MV underground']
    grid_stock_materials.loc[idx[:,material],idx[:,'MV substations']]   = grid_subst_Mv.to_numpy()        * materials_grid_additions.loc[material,'Mv Substation']          # kms * units/km * kg/unit = kg
    grid_stock_materials.loc[idx[:,material],idx[:,'MV transformers']]  = grid_trans_Mv.to_numpy()        * grid_additions.loc['Transformers','Mv'] * materials_grid_additions.loc[material,'Mv Transformer']        # kms * units/km * kg/unit = kg
    grid_stock_materials.loc[idx[:,material],idx[:,'LV overhead']]      = grid_length_Lv_above.to_numpy() * materials_grid.loc[material,'LV overhead']
    grid_stock_materials.loc[idx[:,material],idx[:,'LV underground']]   = grid_length_Lv_under.to_numpy() * materials_grid.loc[material,'LV underground']
    grid_stock_materials.loc[idx[:,material],idx[:,'LV substations']]   = grid_subst_Lv.to_numpy()        * materials_grid_additions.loc[material,'Lv Substation']          # kms * units/km * kg/unit = kg
    grid_stock_materials.loc[idx[:,material],idx[:,'LV transformers']]  = grid_length_Lv_total.to_numpy() * grid_additions.loc['Transformers','Lv'] * materials_grid_additions.loc[material,'Lv Transformer']        # kms * units/km * kg/unit = kg


#%% Inflow Outflow calculations 
from dynamic_stock_model import DynamicStockModel as DSM
first_year_grid = 1926          # UK Electricity supply act - https://www.bbc.com/news/uk-politics-11619751
stdev_mult = 0.214              # standard deviation as a fraction of the mean lifetime applicable to energy equipment (Asset Management for Infrastructure Systems: Energy and Water, Balzer & Schorn 2015)

# In order to calculate inflow & outflow smoothly (without peaks for the initial years), we calculate a historic tail to the stock, by adding a 0 value for first year of operation (=1926), then interpolate values towards 1971
def stock_tail(grid_stock):
    zero_value = [0 for i in range(0,regions)]
    grid_stock.loc[first_year_grid] = zero_value  # set all regions to 0 in the year of initial operation
    grid_stock_new  = grid_stock.reindex(list(range(first_year_grid,endyear+1))).interpolate()
    return grid_stock_new

# call the stock_tail function on all lines, substations & transformers, to add historic stock tail between 1926 & 1971
grid_length_Hv_above_new = stock_tail(grid_length_Hv_above) 
grid_length_Mv_above_new = stock_tail(grid_length_Mv_above) 
grid_length_Lv_above_new = stock_tail(grid_length_Lv_above) 
grid_length_Hv_under_new = stock_tail(grid_length_Hv_under) 
grid_length_Mv_under_new = stock_tail(grid_length_Mv_under) 
grid_length_Lv_under_new = stock_tail(grid_length_Lv_under) 
grid_subst_Hv_new = stock_tail(grid_subst_Hv)
grid_subst_Mv_new = stock_tail(grid_subst_Mv)
grid_subst_Lv_new = stock_tail(grid_subst_Lv)
grid_trans_Hv_new = stock_tail(grid_trans_Hv)
grid_trans_Mv_new = stock_tail(grid_trans_Mv)
grid_trans_Lv_new = stock_tail(grid_trans_Lv)

# Function in which the stock-driven DSM is applied to return (the moving average of the) inflow & outflow for all regions
def inflow_outflow(stock, lifetime):

    initial_year = stock.first_valid_index()
    outflow = pd.DataFrame(index=range(startyear,endyear+1), columns=stock.columns)
    inflow =  pd.DataFrame(index=range(startyear,endyear+1), columns=stock.columns)

    # define mean & standard deviation
    mean_list = [lifetime for i in range(0,len(stock))]   
    stdev_list = [mean_list[i] * stdev_mult for i in range(0,len(stock))]  

    for region in list(stock.columns):
        # define and run the DSM                                                                                            # list with the fixed (=mean) lifetime of grid elements, given for every timestep (1926-2100), needed for the DSM as it allows to change lifetime for different cohort (even though we keep it constant)
        DSMforward = DSM(t = np.arange(0,len(stock[region]),1), s=np.array(stock[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(mean_list), 'StdDev': np.array(stdev_list)})  # definition of the DSM based on a folded normal distribution
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
    
        # sum the outflow by cohort to total outflow & apply a 5yr moving average to return only the 'smoothed' results
        out_o = out_oc.sum(axis=1)
        out_o = np.append(out_o,[out_o[-1],out_o[-1]])  # append last value twice, to get a more accurate moving average
        out_i = np.append(out_i,[out_i[-1],out_i[-1]])  # append last value twice, to get a more accurate moving average
    
        # apply moving average & return only 1971-2100 values
        outflow[region] = pd.Series(out_o[2:], index=list(range(initial_year,2101))).rolling(window=5).mean().loc[list(range(1971,2101))]    # Apply moving average                                                                                                      # sum the outflow by cohort to get the total outflow per year
        inflow[region] = pd.Series(out_i[2:], index=list(range(initial_year,2101))).rolling(window=5).mean().loc[list(range(1971,2101))]
       
    return inflow, outflow

# HV inflow/outflow (in kms or units)
Hv_lines_above_in, Hv_lines_above_out = inflow_outflow(grid_length_Hv_above_new, lifetime_grid_elements.loc['lines'].values[0]) # km
Hv_lines_under_in, Hv_lines_under_out = inflow_outflow(grid_length_Hv_under_new, lifetime_grid_elements.loc['lines'].values[0]) # km
Hv_subst_in, Hv_subst_out = inflow_outflow(grid_subst_Hv_new, lifetime_grid_elements.loc['substations'].values[0])              # nr of units
Hv_trans_in, Hv_trans_out = inflow_outflow(grid_trans_Hv_new, lifetime_grid_elements.loc['transformers'].values[0])             # nr of units

#MV inflow/outflow (in kms or units)
Mv_lines_above_in, Mv_lines_above_out = inflow_outflow(grid_length_Mv_above_new, lifetime_grid_elements.loc['lines'].values[0])  # For now, the same lifetime assumptions for Hv elements are applied to MV & LV
Mv_lines_under_in, Mv_lines_under_out = inflow_outflow(grid_length_Mv_under_new, lifetime_grid_elements.loc['lines'].values[0])
Mv_subst_in, Mv_subst_out = inflow_outflow(grid_subst_Mv_new, lifetime_grid_elements.loc['substations'].values[0])
Mv_trans_in, Mv_trans_out = inflow_outflow(grid_trans_Mv_new, lifetime_grid_elements.loc['transformers'].values[0])

#LV inflow/outflow (in kms or units)
Lv_lines_above_in, Lv_lines_above_out = inflow_outflow(grid_length_Lv_above_new, lifetime_grid_elements.loc['lines'].values[0])
Lv_lines_under_in, Lv_lines_under_out = inflow_outflow(grid_length_Lv_under_new, lifetime_grid_elements.loc['lines'].values[0])
Lv_subst_in, Lv_subst_out = inflow_outflow(grid_subst_Lv_new, lifetime_grid_elements.loc['substations'].values[0])
Lv_trans_in, Lv_trans_out = inflow_outflow(grid_trans_Lv_new, lifetime_grid_elements.loc['transformers'].values[0])

#%% prepare output vaiables

# overall container of stock, inflow & outflow of materials (in kgs)
items = ['lines', 'substations', 'transformers']
flow =  ['stock', 'inflow', 'outflow']
columns = pd.MultiIndex.from_product([region_list, items], names=['regions', 'elements'])
index = pd.MultiIndex.from_product([flow, list(range(startyear,endyear+1)), material_list], names=['flow', 'years', 'materials'])
grid_materials = pd.DataFrame(index=index, columns=columns)

# list used to sum detailed (voltage level specific) stock to total of lines
lines_list = ['HV overhead', 'HV underground', 'MV overhead', 'MV underground', 'LV overhead', 'LV underground']    # lines
subst_list = ['HV substations','MV substations','LV substations']
trans_list = ['HV transformers','MV transformers','LV transformers']

for material in material_list:
    grid_materials.loc[idx['stock', :,material],idx[:,'lines']]         = grid_stock_materials.loc[idx[:,material],idx[:,lines_list]].sum(axis=1, level=0).to_numpy()   # kgs in stock for lines
    grid_materials.loc[idx['stock', :,material],idx[:,'substations']]   = grid_stock_materials.loc[idx[:,material],idx[:,subst_list]].sum(axis=1, level=0).to_numpy()   # kgs in stock for substations
    grid_materials.loc[idx['stock', :,material],idx[:,'transformers']]  = grid_stock_materials.loc[idx[:,material],idx[:,trans_list]].sum(axis=1, level=0).to_numpy()   # kgs in stock for transformers
    grid_materials.loc[idx['inflow', :,material],idx[:,'lines']]        = Hv_lines_above_in.to_numpy() * materials_grid.loc[material,'HV overhead'] + Hv_lines_under_in.to_numpy() * materials_grid.loc[material,'HV underground'] + Mv_lines_above_in.to_numpy() * materials_grid.loc[material,'MV overhead'] + Mv_lines_under_in.to_numpy() * materials_grid.loc[material,'MV underground'] + Lv_lines_above_in.to_numpy() * materials_grid.loc[material,'LV overhead'] + Lv_lines_under_in.to_numpy() * materials_grid.loc[material,'LV underground']
    grid_materials.loc[idx['inflow', :,material],idx[:,'substations']]  = Hv_subst_in.to_numpy() * materials_grid_additions.loc[material,'Hv Substation'] + Mv_subst_in.to_numpy() * materials_grid_additions.loc[material,'Mv Substation'] + Lv_subst_in.to_numpy() * materials_grid_additions.loc[material,'Lv Substation']
    grid_materials.loc[idx['inflow', :,material],idx[:,'transformers']] = Hv_trans_in.to_numpy() * materials_grid_additions.loc[material,'Hv Transformer'] + Mv_trans_in.to_numpy() * materials_grid_additions.loc[material,'Mv Transformer'] + Lv_trans_in.to_numpy() * materials_grid_additions.loc[material,'Lv Transformer']
    grid_materials.loc[idx['outflow', :,material],idx[:,'lines']]        = Hv_lines_above_out.to_numpy() * materials_grid.loc[material,'HV overhead'] + Hv_lines_under_out.to_numpy() * materials_grid.loc[material,'HV underground'] + Mv_lines_above_out.to_numpy() * materials_grid.loc[material,'MV overhead'] + Mv_lines_under_out.to_numpy() * materials_grid.loc[material,'MV underground'] + Lv_lines_above_out.to_numpy() * materials_grid.loc[material,'LV overhead'] + Lv_lines_under_out.to_numpy() * materials_grid.loc[material,'LV underground']
    grid_materials.loc[idx['outflow', :,material],idx[:,'substations']]  = Hv_subst_out.to_numpy() * materials_grid_additions.loc[material,'Hv Substation'] + Mv_subst_out.to_numpy() * materials_grid_additions.loc[material,'Mv Substation'] + Lv_subst_out.to_numpy() * materials_grid_additions.loc[material,'Lv Substation']
    grid_materials.loc[idx['outflow', :,material],idx[:,'transformers']] = Hv_trans_out.to_numpy() * materials_grid_additions.loc[material,'Hv Transformer'] + Mv_trans_out.to_numpy() * materials_grid_additions.loc[material,'Mv Transformer'] + Lv_trans_out.to_numpy() * materials_grid_additions.loc[material,'Lv Transformer']

#%% CSV output files (in kilo-tonnes, kt) - 3 flow types (stock, inflow, outlfow) - 26 Regions - 3 product categories (lines, substations, transformers) - 14 materials (but some are not releveant to the grid, so they are 0)
grid_materials_out = pd.concat([grid_materials], keys=['grid'], names=['category']).stack().stack() # add a descriptor column
grid_materials_out = pd.concat([grid_materials_out], keys=['electricity'], names=['sector'])
grid_materials_out = grid_materials_out.unstack(level=3).reorder_levels([5, 2, 0, 1, 4, 3]) / 1000000   # to kt

grid_materials_out[list(range(2000,2051))].sum(level=[1,2,3,4,5]).to_csv('output\\' + variant + '\\grid_materials_output_kt.csv') # in kt
