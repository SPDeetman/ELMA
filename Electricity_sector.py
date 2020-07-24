# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:59:30 2020
@author: Sebasiaan Deetman (deetman@cml.leidenuniv.nl)

This code is provided for review purposes only, please do not use the code or results without consent of the author

This module is used to calculate the materials involved in the electricity generation capacity and electricity storage capacity
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
import math
import scipy

YOUR_DIR = 'C:\\Users\\Admin\\...'
os.chdir(YOUR_DIR)


from past.builtins import execfile
execfile('read_mym.py')         # module to read in mym (IMAGE output) files 
idx = pd.IndexSlice             # needed for slicing multi-index

scenario = "SSP2"
variant  = "450"                # Select scenario here: 'BL' = SSP2 Baseline; '450' = SSP2 2-degree climate policy scenario (450 refers to the kyoto greenhouse gas concentration limit of 450 ppm CO2-eq units)
path = scenario + "\\" +  scenario + "_" + variant + "\\"

cohorts = 50
startyear = 1971
endyear = 2100
years = endyear - startyear  + 1
switchtime = 1990
vehicles = 25
regions = 26

weibull_shape = 1.89
weibull_scale = 10.3
stdev_mult = 0.214      # multiplier that defines the standard deviation (standard deviation = mean * multiplier)

#%% Read in files. FIRST: IMAGE/TIMER files

# read TIMER installed storage capacity (MWh, reservoir)
storage = read_mym_df(path + 'StorResTot.out')                  #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
storage.drop(storage.iloc[:, -2:], inplace = True, axis = 1)    # drop global total column and empty (27) column

kilometrage = pd.read_csv(path + 'kilometrage.csv', index_col='t')        #kilometrage in kms/yr
# kilometrage is defined untill 2008, fill 2008 values untill 2100 
for row in range(2009,2101):
    kilometrage.loc[row] = kilometrage.loc[2008].values
region_list = list(kilometrage.columns.values)                          # get a list with region names
    
loadfactor_data = read_mym_df(path + 'loadfactor.out') 
loadfactor = loadfactor_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1') # loadfactor for cars (in persons per vehicle)
loadfactor = loadfactor.apply(lambda x: [y if y >= 1 else 1 for y in x])                     # some loadfactor values seem to go below 1, which seems weird, so replace all values below 1 with 1
    
passengerkms_data = read_mym_df(path + 'passengerkms.out')                # passenger kilometers in Tera pkm
indexNames = passengerkms_data[ passengerkms_data['DIM_1'] >= 27 ].index
passengerkms_data.drop(indexNames , inplace=True)
passengerkms = passengerkms_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1')
  
BEV_collist = [22, 23, 24, 25]
PHEV_collist= [21, 20, 19, 18, 17, 16]
vehicleshare_data = read_mym_df(path + 'vehicleshare.out')
vehicleshare_data['battery'] = vehicleshare_data[BEV_collist].sum(axis=1)
vehicleshare_data['PHEV'] = vehicleshare_data[PHEV_collist].sum(axis=1)

# get the regional sum of the BEV & PHEV fraction of the fleet, and replace the column names with the region names
PHEV_share = vehicleshare_data[['time','DIM_1', 'PHEV']].pivot_table(index='time',columns='DIM_1')
BEV_share = vehicleshare_data[['time','DIM_1', 'battery']].pivot_table(index='time',columns='DIM_1')

# before we run any calculations, replace the column names to region names
PHEV_share.columns = region_list
BEV_share.columns = region_list
passengerkms.columns = region_list
loadfactor.columns = region_list
storage.columns = region_list

# Materials in generation capacity (a.o. needed for the names of the generation cap. technologies)
composition_generation = pd.read_csv('Storage_cost\\composition_generation.csv',index_col=0).transpose()  # in gram/MW
gcap_tech_list = list(composition_generation.columns.values)    #list of names of the generation technologies
gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path + 'Gcap.out')
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data, index=['time','DIM_1'], values=list(range(1,29)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap.index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_tech_list

# lifetimes of Gcap tech according to van Vuuren 2006 (PhD Thesis)
gcap_lifetime = pd.read_csv(path + 'LTTechnical.csv', index_col='DIM_1')        
gcap_lifetime.index = gcap_tech_list

# read in the storage share in 2016 according to IEA (Technology perspectives 2017)
storage_IEA = pd.read_csv('Storage_cost\\storage_IEA2016.csv', index_col=0)

# read in the storage costs according to IRENA storage report & other sources in the SI
storage_costs = pd.read_csv('Storage_cost\\storage_cost.csv', index_col=0).transpose()

# read in the assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
storage_malus = pd.read_csv('Storage_cost\\storage_malus.csv', index_col=0).transpose()

#read in the assumptions on the long-term price decline after 2050 (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
storage_ltdecline = pd.Series(pd.read_csv('Storage_cost\\storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

#read in the energy density assumptions (kg/kWh storage capacity)
storage_density = pd.read_csv('Storage_cost\\storage_density_kg_per_kwh.csv',index_col=0).transpose()

#read in the lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
storage_lifetime = pd.read_csv('Storage_cost\\storage_lifetime.csv',index_col=0).transpose()

#read in the material composition assumptions (wt% of total battery weight for various materials, total battery weight is given by the density file above)
storage_materials = pd.read_csv('Storage_cost\\storage_materials.csv',index_col=0).transpose()

# prepare model specific variables by interpolation

# turn index to integer for sorting during the next step
storage_costs.index = storage_costs.index.astype('int64')
storage_malus.index = storage_malus.index.astype('int64')
storage_density.index = storage_density.index.astype('int64')
storage_lifetime.index = storage_lifetime.index.astype('int64')

# to interpolate between 2018 and 2030, first create empty rows (NaN values) 
storage_start = storage_costs.first_valid_index()
storage_end =   storage_costs.last_valid_index()
for i in range(storage_start+1,storage_end):
    storage_costs = storage_costs.append(pd.Series(name=i))
    storage_malus = storage_malus.append(pd.Series(name=i))         # mind: the malus needs to be defined for the same years as the cost indications
    storage_density = storage_density.append(pd.Series(name=i))     # mind: the density needs to be defined for the same years as the cost indications
    storage_lifetime = storage_lifetime.append(pd.Series(name=i))   # mind: the lifetime needs to be defined for the same years as the cost indications
    
# then, do the actual interpolation on the sorted dataframes                                                    
storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)
storage_malus_interpol = storage_malus.sort_index(axis=0).interpolate(axis=0)
storage_density_interpol = storage_density.sort_index(axis=0).interpolate(axis=0)  # density calculation continue with the material calculations
storage_lifetime_interpol = storage_lifetime.sort_index(axis=0).interpolate(axis=0)  # lifetime calculation continue with the material calculations

# fix the energy density (kg/kwh) of storage technologies after 2030
for year in range(2030+1,endyear+1):
    storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.last_valid_index()], name=year))

# assumed fixed energy densities before 2018
for year in reversed(range(switchtime,storage_start)):
    storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.first_valid_index()], name=year)).sort_index(axis=0)


#%% 0) Before we start the calculations we define the general functions used in multiple parts of the code

# ----------------- |||| Loop to derive stock share from total stock and market (inflow) share \\\\\ ----------------------------------------------------------

# First the lifetime of storage technologies needs to be defined over time, before running the dynamic stock function
# fix the mean lifetime of storage technologies after 2030 & before 2018
for year in range(2030+1,endyear+1):
    storage_lifetime_interpol = storage_lifetime_interpol.append(pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()], name=year))

for year in reversed(range(startyear,storage_start)):
    storage_lifetime_interpol = storage_lifetime_interpol.append(pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()], name=year)).sort_index(axis=0)

# drop the PHS from the interpolated lifetime frame, as the PHS is calculated separately
storage_lifetime_interpol = storage_lifetime_interpol.drop(columns=['PHS'])

# Here, we define the market share of the stock based on a pre-calculation with several steps: 1) use Global total stock development, the market shares of the inflow and technology specific lifetimes to derive the shares in the stock, assuming that pre-1990 100% of the stock was determined by Lead-acid batteries. 2) Then, apply the global stock-related market shares to disaggregate the stock to technologies in all regions (assuming that battery markets are global markets)
# As the development of the total stock of dedicated electricity storage is known, but we don't know the inflow, and only the market share related to the inflow we need to calculate the inflow first. 

# first we define the survival of the 1990 stock (assumed 100% Lead-acid, for each cohort in 1990)
pre_time = 20                           # the years required for the pre-calculation of the Lead-acid stock
cohorts = endyear-switchtime            # nr. of cohorts in the stock calculations (after switchtime)
timeframe = np.arange(0,cohorts+1)      # timeframe for the pre-calcluation
pre_year_list = list(range(switchtime-pre_time,switchtime+1))   # list of years to pre-calculate the Lead-acid stock for

def stock_share_calc(stock, market_share, init_tech, techlist):
    
    #define new dataframes
    stock_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(switchtime,endyear+1)]), columns=pd.MultiIndex.from_product([techlist, range(switchtime-pre_time,endyear+1)])) # year, by year, by tech
    inflow_by_tech = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(switchtime,endyear+1)]), columns=techlist)
    outflow_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(switchtime,endyear+1)]), columns=pd.MultiIndex.from_product([techlist, range(switchtime-pre_time,endyear+1)])) # year by year by tech
    
    #specify lifetime & other settings
    mean = storage_lifetime_interpol.loc[switchtime,init_tech] # select the mean lifetime for Lead-acid batteries in 1990
    stdev = mean * stdev_mult               # we use thesame standard-deviation as for generation technologies, given that these apply to 'energy systems' more generally
    survival_init = scipy.stats.foldnorm.sf(timeframe, mean/stdev, 0, scale=stdev)
    techlist_new = techlist
    techlist_new.remove(init_tech)          # techlist without Lead-acid (or other init_tech)
    
    # actual inflow & outflow calculations, this bit takes long!
    # loop over regions, technologies and years to calculate the inflow, stock & outflow of storage technologies, given their share of the inflow.
    for region in stock.columns:
        
        # pre-calculate the stock by cohort of the initial stock of Lead-acid
        multiplier_pre = stock.loc[switchtime,region]/survival_init.sum()                                    # the stock is subdivided by the previous cohorts according to the survival function (only allowed when assuming steady stock inflow) 
        
        #pre-calculate the stock as lists (for efficiency)
        initial_stock_years = [np.flip(survival_init[0:pre_time+1]) * multiplier_pre]
            
        for year in range(1, (endyear-switchtime)+1):                                                                   # then fill the columns with the remaining fractions
            initial_stock_years.append(initial_stock_years[0] * survival_init[year])
    
        stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]] = initial_stock_years       # fill the stock dataframe according to the pre-calculated stock 
        outflow_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]] = stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]].shift(1, axis=0) - stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]]
    
        # set the other stock cohorts to zero
        stock_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        outflow_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        inflow_by_tech.loc[idx[region, switchtime], techlist_new] = 0                                                   # inflow of other technologies in 1990 = 0
        
        # except for outflow and inflow in 1990 (switchtime), which can be pre-calculated for Deep-cycle Lead Acid (@ steady state inflow, inflow = outflow = stock/lifetime)
        outflow_cohorts.loc[idx[region, switchtime], idx[init_tech,:]] = outflow_cohorts.loc[idx[region, switchtime+1], idx[init_tech,:]]     # given the assumption of steady state inflow (pre switchtime), we can determine that the outflow is the same in switchyear as in switchyear+1                                        
        inflow_by_tech.loc[idx[region, switchtime], init_tech] =   stock.loc[switchtime,region]/mean                                          # given the assumption of steady state inflow (pre switchtime), we can determine the inflow to be the same as the outflow at a value of stock/avg. lifetime                                                          
        
        # From switchtime onwards, define a stock-driven model with a known market share (by tech) of the new inflow 
        for year in range(switchtime+1,endyear+1):
            
            # calculate the remaining stock as the sum of all cohorts in a year, for each technology
            remaining_stock = 0 # reset remaining stock
            for tech in inflow_by_tech.columns:
                remaining_stock += stock_cohorts.loc[idx[region,year],idx[tech,:]].sum()
               
            # total inflow required (= required stock - remaining stock);    
            inflow = max(0, stock.loc[year, region] - remaining_stock)   # max 0 avoids negative inflow, but allows for idle stock surplus in case the size of the required stock is declining more rapidly than it's natural decay
            
            stock_cohorts_list = []
                   
            # enter the new inflow & apply the survival rate, which is different for each technology, so calculate the surviving fraction in stock for each technology  
            for tech in inflow_by_tech.columns:
                # apply the known market share to the inflow
                inflow_by_tech.loc[idx[region,year],tech] = inflow * market_share.loc[year,tech]
                # first calculate the survival (based on lifetimes specific to the year of inflow)
                survival = scipy.stats.foldnorm.sf(np.arange(0,(endyear+1)-year), storage_lifetime_interpol.loc[year,tech]/(storage_lifetime_interpol.loc[year,tech]*0.2), 0, scale=storage_lifetime_interpol.loc[year,tech]*0.2)           
                # then apply the survival to the inflow in current cohort, both the inflow & the survival are entered into the stock_cohort dataframe in 1 step
                stock_cohorts_list.append(inflow_by_tech.loc[idx[region,year],tech]  *  survival)
                
            stock_cohorts.loc[idx[region,list(range(year,endyear+1))],idx[:,year]] = list(map(list, zip(*stock_cohorts_list)))        
        
    # separate the outflow (by cohort) calculation (separate shift calculation for each region & tech is MUCH more efficient than including it in additional loop over years)
    # calculate the outflow by cohort based on the stock by cohort that was just calculated
    for region in stock.columns:
        for tech in inflow_by_tech.columns:
            outflow_cohorts.loc[idx[region,:],idx[tech,:]] = stock_cohorts.loc[idx[region,:],idx[tech,:]].shift(1,axis=0) - stock_cohorts.loc[idx[region,:],idx[tech,:]]

    return inflow_by_tech, stock_cohorts, outflow_cohorts

#%% 1) start with materials in generation capacity (this is the easiest, as the stock AND the new capacity is pre-calculated in TIMER, based on fixed lifetime assumptions, we only add the outflow, based on a fixed lifetime DSM and the same lifetimes as in TIMER)
#   1.1) Apply the DSM to find inflow & outflow of Generation capacity
    
from dynamic_stock_model import DynamicStockModel as DSM

first_year_grid = 1926          # UK Electricity supply act - based on Gaby Hornsby - BBC 2010: https://www.bbc.com/news/uk-politics-11619751   
      
# In order to calculate inflow & outflow smoothly (without peaks for the initial years), we calculate a historic tail to the stock, by adding a 0 value for first year of operation (=1926), then interpolate values towards 1971
def stock_tail(stock):
    zero_value = [0 for i in range(0,regions)]
    stock_used = pd.DataFrame(stock).reindex_like(stock)
    stock_used.loc[first_year_grid] = zero_value  # set all regions to 0 in the year of initial operation
    stock_new  = stock_used.reindex(list(range(first_year_grid,endyear+1))).interpolate()
    return stock_new

# first define a Function in which the stock-driven DSM is applied to return (the moving average of the) inflow & outflow for all regions
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
       
    return inflow.stack(), outflow.stack()

# Calculate the historic tail to the Gcap (stock) 
gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,endyear+1), region_list], names=['years', 'regions']), columns=gcap.columns)

for tech in gcap_tech_list:
    gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1)).stack()

# then apply the Dynamic Stock Model to find inflow & outflow (5yr moving average)
gcap_in = pd.DataFrame(index=gcap.index, columns=gcap.columns)
gcap_out = pd.DataFrame(index=gcap.index, columns=gcap.columns)

for tech in gcap_tech_list:
    gcap_in.loc[idx[:,:],tech], gcap_out.loc[idx[:,:],tech] = inflow_outflow(gcap_new.loc[idx[:,:],tech].unstack(level=1), gcap_lifetime.loc[tech].values[0])

# 1.2) Materials in generation capacity 
# new dataframe containing the total weight of materials (in grams) in the electricity generation technologies
columns = pd.MultiIndex.from_product([gcap.columns, gcap_material_list], names=['technologies','materials'])
gcap_materials_stock = pd.DataFrame(index=gcap.index, columns=columns)
gcap_materials_inflow = pd.DataFrame(index=gcap.index, columns=columns)
gcap_materials_outflow = pd.DataFrame(index=gcap.index, columns=columns)

for tech in gcap_tech_list:
    for material in gcap_material_list:
        for region in region_list:
            gcap_materials_stock.loc[idx[:,region],idx[tech,material]]  = gcap.loc[idx[:,region],tech].to_numpy() * composition_generation.loc[material, tech]
            gcap_materials_inflow.loc[idx[:,region],idx[tech,material]] = gcap_in.loc[idx[:,region],tech].to_numpy() * composition_generation.loc[material, tech]
            gcap_materials_outflow.loc[idx[:,region],idx[tech,material]] = gcap_out.loc[idx[:,region],tech].to_numpy() * composition_generation.loc[material, tech]

#prepare variables on materials in generation capacity (in gram) for output in csv
gcap_materials_stock = gcap_materials_stock.stack().stack().unstack(level=0)        # use years as column names
gcap_materials_stock = pd.concat([gcap_materials_stock], keys=['stock'], names=['flow'])

gcap_materials_inflow = gcap_materials_inflow.stack().stack().unstack(level=0)        # use years as column names
gcap_materials_inflow = pd.concat([gcap_materials_inflow], keys=['inflow'], names=['flow'])

gcap_materials_outflow = gcap_materials_outflow.stack().stack().unstack(level=0)        # use years as column names
gcap_materials_outflow = pd.concat([gcap_materials_outflow], keys=['outflow'], names=['flow'])

gcap_materials_all = pd.concat([gcap_materials_stock, gcap_materials_inflow, gcap_materials_outflow])
gcap_materials_all = pd.concat([gcap_materials_all], keys=['electricity'], names=['sector'])
gcap_materials_all = pd.concat([gcap_materials_all], keys=['generation'], names=['category'])
gcap_materials_all = gcap_materials_all.reorder_levels([3, 2, 1, 0, 5, 4]) / 1000000000   # gram to kt

gcap_materials_all[list(range(2000,2051))].sum(level=[1,2,3,4,5]).to_csv('output\\' + variant + '\\gcap_materials_output_kt.csv') # in kt

#%% 2) Then, determine the market share of the storage capacity using a multi-nomial logit function

# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
decline_used = decline*storage_ltdecline

storage_costs_new = storage_costs_interpol * storage_malus_interpol

# calculate the development from 2030 to 2050 (using annual price decline)
for year in range(storage_end+1,2050+1):
    storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))

# for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
for year in reversed(range(startyear,storage_start)):
    storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid']        # restore the exception (set to constant 2018 values)

# Multinomial Logit function, assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)
def MNLogit(df, logitpar):
    new_dataframe = pd.DataFrame(index=df.index, columns=df.columns)
    for year in range(df.index[0],df.index[-1]+1): #from first to last year
        yearsum = 0
        for column in df.columns:
            yearsum += math.exp(logitpar * df.loc[year,column]) # calculate the sum of the prices
        for column in df.columns:
            new_dataframe.loc[year,column] = math.exp(logitpar * df.loc[year,column])/yearsum
    return new_dataframe                                                                            # the retuned dataframe contains the market shares

# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(storage_costs_new, -0.2)

# fix the market share of storage technologies after 2050
for year in range(2050+1,endyear+1):
    storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))

#%% 2) Second Section: Availability of storage in vehicles & pumped hydro. 
# 2.1) starting with Vehicles (total nr. of)

# BEV & PHEV vehicle stats
BEV_capacity  = 59.6    #kWh current battery capacity of full electric vehicles, see current_specs.xlsx
PHEV_capacity = 11.2    #kWh current battery capacity of plugin electric vehicles, see current_specs.xlsx
capacity_usable_PHEV = 0.05 # 5% of capacity of PHEV is usable as storage
capacity_usable_BEV  = 0.10 # 10% of capacity of BEVs is usable as storage

vehicle_kms = passengerkms * 1000000000000 / loadfactor        # conversion from tera-Tkms  
vehicles_all = vehicle_kms / kilometrage

vehicles_PHEV = vehicles_all * PHEV_share
vehicles_BEV = vehicles_all * BEV_share
vehicles_EV = vehicles_PHEV + vehicles_BEV

# For the availability of vehicle store capacity we apply the assumption of fixed weight,
# So we first need to know the average density of the stock of EV batteries
# to get there we need to know the share of the technologies in the stock (based on lifetime & share in the inflow/purchases)
# to get there, we first need to know the market share of new stock additions (=inflow/purchases)

#First we calculate the share of the inflow using only a few of the technologies in the storage market share
#The selection represents only the batteries that are suitable for EV & mobile applications
EV_battery_list = ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air']
#normalize the selection of market shares, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = pd.DataFrame().reindex_like(storage_market_share[EV_battery_list])

for year in list(range(startyear,endyear+1)):
    for tech in EV_battery_list:
        market_share_EVs.loc[year, tech] = storage_market_share[EV_battery_list].loc[year,tech] /storage_market_share[EV_battery_list].loc[year].sum()

# then we use that market share in combination with the stock developments to derive the stock share 
# Here we use the vehcile stock (number of cars) as a proxy for the development of the battery stock (given that we're calculating the actual battery stock still, and just need to account for the dynamics of purchases to derive te stock share here) 
EV_inflow_by_tech, EV_stock_cohorts, EV_outflow_cohorts = stock_share_calc(vehicles_EV, market_share_EVs, 'NiMH', ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air'])

EV_stock = EV_stock_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0)
EV_storage_stock_abs = EV_stock.sum(axis=0, level=1)                           # sum over all regions to get the global share of the stock
EV_stock_total = EV_stock.sum(axis=1).unstack(level=0)                            # total stock by region
EV_storage_stock_share = pd.DataFrame(index=EV_storage_stock_abs.index, columns=EV_storage_stock_abs.columns)

# sum to the global share of the battery technologies in stock
for tech in EV_storage_stock_abs.columns:
    EV_storage_stock_share.loc[:,tech] = EV_storage_stock_abs.loc[:,tech].div(EV_storage_stock_abs.sum(axis=1))

#The global share of the battery technologies in stock is then used to derive the (weihgted) average density (kg/kWh)
weighted_average_density = EV_storage_stock_share.mul(storage_density_interpol[EV_battery_list]).sum(axis=1)

# assumed fixed energy densities before 1990 (=NiMH)
add = pd.Series(weighted_average_density[weighted_average_density.first_valid_index()], index=list(range(startyear,1990)))
weighted_average_density = weighted_average_density.append(add).sort_index(axis=0)

# With a pre-determined battery capacity in 2018, we assume an increasing capacity (as an effect of an increased density) based on a fixed weight assumption
BEV_dynamic_capacity = (weighted_average_density[2018] * BEV_capacity) / weighted_average_density
PHEV_dynamic_capacity = (weighted_average_density[2018] * PHEV_capacity) / weighted_average_density

# Define the V2G settings over time (assuming slow adoption to the maximum usable capacity)
year_v2g_start = 2025-1 # -1 leads to usable capacity in 2025
year_full_capacity = 2040
v2g_usable = pd.Series(index=list(range(startyear,endyear+1)), name='years')    #fraction of the cars ready and willing to use v2g
for year in list(range(startyear,endyear+1)):
    if (year <= year_v2g_start):
        v2g_usable[year] = 0
    elif (year >= year_full_capacity):
        v2g_usable[year] = 1
v2g_usable = v2g_usable.interpolate()

#total capacity per car connected to v2g (in kWh)
max_capacity_BEV = v2g_usable * capacity_usable_BEV  
max_capacity_PHEV = v2g_usable * capacity_usable_PHEV   

#usable capacity per car connected to v2g (in kWh)
usable_capacity_BEV = max_capacity_BEV.mul(BEV_dynamic_capacity)     
usable_capacity_PHEV = max_capacity_PHEV.mul(PHEV_dynamic_capacity)  

# Vehicle storage in MWh
storage_BEV = storage_PHEV = pd.DataFrame().reindex_like(vehicles_BEV)
for region in region_list:
    storage_PHEV[region] = vehicles_PHEV[region].mul(usable_capacity_PHEV) /1000
    storage_BEV[region] = vehicles_BEV[region].mul(usable_capacity_BEV) /1000

storage_vehicles = storage_BEV + storage_PHEV

#%% 2.2) Take the TIMER Hydro-dam capacity (MW) & compare it to Pumped hydro capacity (MW) projections from the International Hydropower Association
Gcap_hydro = gcap_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1')   #Hydro dam capacity (power, in MW)
Gcap_hydro.columns = region_list

#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
storage_power = read_mym_df(path + 'StorCapTot.out')                  
storage_power.drop(storage_power.iloc[:, -2:], inplace = True, axis = 1)    
storage_power.columns = region_list

#Disaggregate the Pumped hydro-storgae projections to 26 IMAGE regions according to the relative Hydro-dam power capacity (also MW) within 5 regions reported by the IHS (international Hydropwer Association)
phs_projections = pd.read_csv(path + 'PHS.csv', index_col='t')                                  # pumped hydro storage capacity (MW)
phs_regions = [[10,11],[19],[1],[22],[0,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,20,21,23,24,25]]   # subregions in IHS data for Europe, China, US, Japan, RoW, MIND: region refers to IMAGE region MINUS 1
phs_projections_IMAGE = pd.DataFrame(index=Gcap_hydro.index, columns=Gcap_hydro.columns)        # empty dataframe

for column in range(0,len(phs_regions)):
    sum_data = Gcap_hydro.iloc[:,phs_regions[column]].sum(axis=1)                               # first, get the sum of all hydropower in the IHS regions (to divide over in second step)
    for region in range(0,regions):
        if region in phs_regions[column]:
            phs_projections_IMAGE.iloc[:,region] = phs_projections.iloc[:,column] * (Gcap_hydro.iloc[:,region]/sum_data)

# Then fill the years after 2030 (end of IHS projections) according to the Gcap annual growth rate (assuming a fixed percentage of Hydro dams will be built with Pumped hydro capabilities after )
phs_projections_IMAGE.loc[2030:2100] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:2100]/Gcap_hydro.loc[2030])

# Calculate the fractions of the storage capacity that is provided through pumped hydro-storage, electric vehicles or other storage (larger than 1 means the capacity superseeds the demand for energy storage, in terms of power in MW or enery in MWh) 
phs_storage_fraction = phs_projections_IMAGE.divide(storage_power)              # both phs & storage_power here are expressed in MW
evs_storage_fraction = storage_vehicles.divide(storage)                         # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
oth_storage_fraction = 1 - (phs_storage_fraction + evs_storage_fraction)     
oth_storage_fraction = oth_storage_fraction.where(oth_storage_fraction >= 0, 0) # the other fraction is the fraction of the storage demand that is not satisfied with pumped hydro storage or electric vehicle storage, i.e. the 'dedicated' energy storage

# absolute storage capacity (MWh)
phs_storage = phs_storage_fraction * storage
evs_storage = evs_storage_fraction * storage
oth_storage = oth_storage_fraction * storage

#output for Main text figure 2 (storage reservoir, in MWh for 3 storage types)
storage_out_phs = pd.concat([phs_storage], keys=['phs'], names=['type']) 
storage_out_evs = pd.concat([evs_storage], keys=['evs'], names=['type']) 
storage_out_oth = pd.concat([oth_storage], keys=['oth'], names=['type']) 
storage_out = pd.concat([storage_out_phs, storage_out_evs, storage_out_oth])
storage_out.to_csv('output\\' + variant + '\\storage_by_type_MWh.csv')        # in MWh

#derive inflow & outflow (in MWh) for PHS, for later use in the material calculations 
phs_storage_stock = stock_tail(phs_storage)
phs_storage_inflow = inflow_outflow(phs_storage_stock, storage_lifetime['PHS'][2018])[0].unstack()     # PHS lifetime is fixed at 60 yrs anyway so, we simply select 1 value
phs_storage_outflow = inflow_outflow(phs_storage_stock, storage_lifetime['PHS'][2018])[1].unstack()  # outflow = second of tuple

#%% 3) Calculate the materials in dedicated electricity storage capacity

import matplotlib.pyplot as plt

lion_tech   = storage_market_share.columns[5:10]    # Lithium-ion technologies (including lifepo4 & LTO)
flow_tech   = storage_market_share.columns[10:12]   # flow batteries: Zinc-Bromide & Vanadium Redox
salt_tech   = storage_market_share.columns[12:14]   # molten salt batteries: Sodium sulfur & ZEBRA
advl_tech   = storage_market_share.columns[14:18]   # Advanced lithium technologies (Li-S, Li-ceramic & Li-air)
othr_tech   = storage_market_share.columns[3:5]     # NiMH & Lead-Acid
legend_elements3 = ['Flywheels','Compressed Air', 'Lithium-ion', 'Advanced Li', 'Flow Batteries', 'Molten Salt batteries', 'Hydrogen', 'Other']
color_set = ("#a82c41", "#6a26d1", "#30942b", "#5c56d6", "#edce77", "#f0ba3c", "#babab8", "#4287f5")

#  3.1) Figure on storage shares, annual development of the market share (stacked area plot @ 100%)
y = np.vstack([np.array(storage_market_share['Flywheel'].astype(float)), np.array(storage_market_share['Compressed Air'].astype(float)), np.array(storage_market_share[lion_tech].sum(axis=1).astype(float)), np.array(storage_market_share[advl_tech].sum(axis=1).astype(float)), np.array(storage_market_share[flow_tech].sum(axis=1).astype(float)), np.array(storage_market_share[salt_tech].sum(axis=1).astype(float)),  np.array(storage_market_share['Hydrogen FC'].astype(float)), np.array(storage_market_share[othr_tech].sum(axis=1).astype(float))])
fig, ax = plt.subplots(figsize=(14, 10))
ax.margins(0)
ax.tick_params(labelsize=16)
ax.stackplot(np.array(storage_market_share.index[19:90]), np.flip(y[:,19:90], axis=0), labels=legend_elements3, colors=color_set)
ax.legend(loc='upper left')
plt.legend(legend_elements3, loc=8, bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=14)
plt.tight_layout(pad=10)
plt.savefig('output\\' + variant + '\\graphs\\market_share_storage.jpg', dpi=600)

#%% 3.2) apply the dedicated electricity storage market shares to the demand for dedicated (other) storage to find storage by type and by region

import sys 
sys.path.append(YOUR_DIR)
from dynamic_stock_model import DynamicStockModel as DSM

#%% -------- INflow & OUTflow calculations ---------


# then, calculate the market share of technologies in the stock (for region & by year), also a global total market share of the stock is calculated to compare to the inflow market share 
inflow_by_tech, stock_cohorts, outflow_cohorts = stock_share_calc(oth_storage, storage_market_share, 'Deep-cycle Lead-Acid', list(storage_lifetime_interpol.columns)) # run the function that calculates stock shares from total stock & inflow shares
stock = stock_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0)
storage_stock_abs = stock.sum(axis=0, level=1)                              # sum over all regions to get the global share of the stock
stock_total = stock.sum(axis=1).unstack(level=0)                            # total stock by region
storage_stock_share = pd.DataFrame(index=storage_stock_abs.index, columns=storage_stock_abs.columns)

for tech in storage_stock_abs.columns:
    storage_stock_share.loc[:,tech] = storage_stock_abs.loc[:,tech].div(storage_stock_abs.sum(axis=1))

# then, the sum of the global inflow & outflow is calculated for comparison in the figures below
outflow = outflow_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0)
outflow_total = outflow.sum(axis=1).unstack(level=0)                        # total by regions

inflow_total = inflow_by_tech.sum(axis=1).unstack(level=0)

#%% 4) ---------- Material calculations ---------------

# define empty dataframe for the weight of storage technolgies (in kg)
i_storage_weight = pd.DataFrame().reindex_like(stock)            
o_storage_weight = pd.DataFrame().reindex_like(stock)           
s_storage_weight = pd.DataFrame().reindex_like(stock)           

# calculate the total weight of the inflow of storage technologies in kg based on the (CHANGEING!) energy density (kg/kWh) and storage demand (MWh)
for region in oth_storage.columns:
    inflow_multiplier = inflow_by_tech.loc[idx[region,:],:]
    inflow_multiplier.index = inflow_multiplier.index.droplevel(0)
    i_storage_weight.loc[idx[region,:],:] = storage_density_interpol.mul(inflow_multiplier).values * 1000       # * 1000 because density is in kg/kWh and storage is in MWh

# calculate the weight of the stock and the outflow (in kg)
# this one's tricky, the outflow and stock are calculated by year AND cohort, because the total weight is represented by the sum of the weight of each cohort. As the density (kg/kWh) is different for every cohort (battery weight changes over time, older batteries are heavier), the total outflow FOR EACH YEAR is the sumproduct of the density and the outflow by cohort.
for region in oth_storage.columns:
    for tech in storage_density_interpol.columns:
        for year in storage_density_interpol.index:
            
            # the first series of the sumproduct (the storage capacity in MWh) has a multi-level index, so the second level needs to be removed before it can be multiplied
            series_mwh_o = outflow_cohorts.loc[idx[region,year],idx[tech,:]]
            series_mwh_o.index = series_mwh_o.index.droplevel(0)                
            o_storage_weight.loc[idx[region,year],tech] = series_mwh_o.mul(storage_density_interpol[tech] * 1000).sum()       # * 1000 because density is in kg/kWh and storage is in MWh

            # same for stock
            series_mwh_s = stock_cohorts.loc[idx[region,year],idx[tech,:]]
            series_mwh_s.index = series_mwh_s.index.droplevel(0)                
            s_storage_weight.loc[idx[region,year],tech] = series_mwh_s.mul(storage_density_interpol[tech] * 1000).sum()       # * 1000 because density is in kg/kWh and storage is in MWh

# Add the weight (in kg) of the PHS storage component, which was calculated before (given in MWh) - 
PHS_kg_perkWh = 26.8            # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 
i_storage_weight.loc[idx[:,:],'PHS'] = phs_storage_inflow.stack().reorder_levels([1,0]) * PHS_kg_perkWh * 1000
s_storage_weight.loc[idx[:,:],'PHS'] = phs_storage.stack().reorder_levels([1,0]) * PHS_kg_perkWh * 1000
o_storage_weight.loc[idx[:,:],'PHS'] = phs_storage_outflow.stack().reorder_levels([1,0]) * PHS_kg_perkWh * 1000

# Now, based on the total weight of the storage by technology, calculate the weight of the materials, using a constant material fraction per technology (storage_materials)

# empty dataframe 
index = pd.MultiIndex.from_product([storage_density_interpol.index, storage_materials.index], names=['years', 'materials'])
columns = pd.MultiIndex.from_product([oth_storage.columns, s_storage_weight.columns], names=['regions', 'technologies'])
i_materials_by_tech = pd.DataFrame(np.zeros((len(storage_stock_share.index), len(storage_materials.index),len(oth_storage.columns),len(s_storage_weight.columns),)).reshape(len(storage_stock_share.index)*len(storage_materials.index),len(oth_storage.columns)*len(s_storage_weight.columns)), index=index, columns=columns)
s_materials_by_tech = pd.DataFrame().reindex_like(i_materials_by_tech)
o_materials_by_tech = pd.DataFrame().reindex_like(i_materials_by_tech)

# materials in kt by region & by type (4dims: index=[year, material], columns=[regions & tech])
for region in oth_storage.columns:
    for material in storage_materials.index:
        for tech in storage_materials.columns:
            i_materials_by_tech.loc[idx[:,material],idx[region,tech]] = i_storage_weight.loc[idx[region,:],tech].mul(storage_materials.loc[material,tech]).values / 1000000 # kg to kt
            s_materials_by_tech.loc[idx[:,material],idx[region,tech]] = s_storage_weight.loc[idx[region,:],tech].mul(storage_materials.loc[material,tech]).values / 1000000 # kg to kt
            o_materials_by_tech.loc[idx[:,material],idx[region,tech]] = o_storage_weight.loc[idx[region,:],tech].mul(storage_materials.loc[material,tech]).values / 1000000 # kg to kt
                        
# Test: For compressed air, the sum of the material fractions is 1, so we use it to test if the sum of the materials in compressed air is similar to the total weight
if (abs(i_materials_by_tech.loc[idx[2000,:],idx['N.Africa','Compressed Air']].sum() - (i_storage_weight.loc[idx['N.Africa',2000],'Compressed Air'] /1000000)) > 0.0000001): print("WARNING: SOMETHING'S WRONG!")

#%% Export data to excel (in kt)

# full storage dataset (by technology)
i_materials_by_tech_out = i_materials_by_tech.unstack(level=1).transpose().reset_index(inplace=False)
i_materials_by_tech_out.insert(1, 'flow', 'inflow')     # add a 'flow' column, 
s_materials_by_tech_out = s_materials_by_tech.unstack(level=1).transpose().reset_index(inplace=False)
s_materials_by_tech_out.insert(1, 'flow', 'stock')      # add a 'flow' column
o_materials_by_tech_out = o_materials_by_tech.unstack(level=1).transpose().reset_index(inplace=False) 
o_materials_by_tech_out.insert(1, 'flow', 'outflow')    # add a 'flow' column

#combine stock, inflow and outflow for output & export to excel (in kt)
output_by_tech = pd.concat([s_materials_by_tech_out, i_materials_by_tech_out, o_materials_by_tech_out]) 
output_by_tech.insert(2, 'category', 'storage')      # add a 'category' column
output_by_tech.insert(2, 'sector', 'electricity')    # add a 'sector' column

output_by_tech = output_by_tech.set_index(['regions', 'flow', 'sector','category','technologies','materials'])
output_by_tech[list(range(2000,2051))].sum(level=[1,2,3,4,5]).to_csv('output\\' + variant + '\\stor_materials_output_kt.csv', index=True) # in kt
