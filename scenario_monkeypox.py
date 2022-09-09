
#import sys
import os

# Standard imports
import random
import sys

random.seed(4475772444933854010)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime as dt
import time
from datetime import timedelta
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import dweibull
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import math
from functools import reduce
import matplotlib.pyplot as plt

# For time-series modeling
from seir_monkeypox import LogNormalSEIR
from helpers_monkeypox import *
np.set_printoptions(threshold=np.inf)

#sys.path.append(os.pardir)
output_path = "./"    
data_file = 'cases-by-day.csv'    
actual_cases = pd.read_csv(data_file, parse_dates=["diagnosis_date"]).set_index("diagnosis_date")
region = 'nyc'


def plot_results(sim_results, percent_diagnosed = 1):
    #sim_results = sim_results.shift(periods=10)
    plt.figure(figsize=(12,5))
    case_df = actual_cases.loc[actual_cases['incomplete'] == '-',:].copy()
    case_df.loc[:,'sim_counts'] = sim_results * percent_diagnosed
    sim_results = sim_results * percent_diagnosed
    plt.plot(sim_results.index, sim_results, label="Simulated Cases")
    plt.scatter(case_df.index, case_df['count'], label="Reported Cases", c="r" )
    plt.ylim([0,250])
    plt.xticks(rotation = 15)
    plt.ylabel("Daily New Monkeypox Cases")
    plt.legend()
    plt.savefig("BestFit.png")
    plt.figure(figsize=(12,5))
    plt.plot(sim_results.index, sim_results.cumsum(), label="Simulated Cases", c="#7189BF")
    plt.scatter(case_df.index, case_df['count'].cumsum(), label="Reported Cases", c="#DF7599" )
    plt.xticks(rotation = 15)
    plt.ylabel("Cumulative Monkeypox Cases")
    plt.legend()
    plt.savefig("BestFitCumulative.png")

def calc_cost(sim_results, percent_diagnosed = 1):
    #sim_results = sim_results.shift(periods=10) # add lag
    case_df = actual_cases.loc[actual_cases['incomplete'] == '-',:].copy()
    case_df.loc[:,'sim_counts'] = sim_results * percent_diagnosed
    case_df['err'] = case_df["count"] - case_df['sim_counts']
    case_df['se'] = case_df['err'] * case_df['err']
    return case_df.loc[:,"se"].mean()
    #Create output csv files 

def save_output_to_csv (new_infections, output_combined, append=''):
    output_dir = os.path.join(output_path,"simulations/")
    os.makedirs(os.path.join(output_path,"simulations/"), exist_ok=True)
        
    new_infections.to_csv(os.path.join(output_dir, "nyc_new_infections_%s" % append)) #Number of new daily infections only 
    output_combined.to_csv(os.path.join(output_dir, "nyc_combined_%s" % append)) #Number for all compartments
    
def samples_2_df ( samples, time):
    outputS = samples[:, 0, 1:]
    outputE = samples[:, 1, 1:] 
    outputI = samples[:, 2, 1:] 
    outputR = samples[:, 3, 1:] 
    output_vacc_actual = samples[:, 4, 1:]
    output_newinfections = samples[:, 5, 1:]
    
    # Convert to dataframe for clarity
    output_newinfections = pd.DataFrame(output_newinfections.T,
                            columns=["s{}".format(i) for i in range(output_newinfections.shape[0])],
                            index=time[1:]).median(axis=1)

    outputS = pd.DataFrame(outputS.T,
                            columns=["s{}".format(i) for i in range(outputS.shape[0])],
                            index=time[1:]).median(axis=1)                                           

    outputE = pd.DataFrame(outputE.T,
                            columns=["s{}".format(i) for i in range(outputE.shape[0])],
                            index=time[1:]).median(axis=1)                                              

    outputI = pd.DataFrame(outputI.T,
                            columns=["s{}".format(i) for i in range(outputI.shape[0])],
                            index=time[1:]).median(axis=1)  

    outputR = pd.DataFrame(outputR.T,
                            columns=["s{}".format(i) for i in range(outputR.shape[0])],
                            index=time[1:]).median(axis=1)  
    
    output_vacc_actual = pd.DataFrame(output_vacc_actual.T,
                                columns=["s{}".format(i) for i in range(output_vacc_actual.shape[0])],
                                index=time[1:]).median(axis=1)
    
    output_combined = pd.concat([outputS, outputE, outputI, outputR, output_vacc_actual, output_newinfections], axis=1).reindex(outputS.index)
    return output_newinfections, output_combined

def sim_monkeypox(immune_num_effects, 
                    social_distance_start_date = "2022-06-15", 
                    social_distance_effect =  0.95, 
                    D_e = 10, 
                    D_i = 21, 
                    monkeypox_base_R0 = 2.0, 
                    population = 70_180, 
                    g_i = 3, 
                    monkeypox_sig_eps=0.7224738559305864,
                    write_output_to_csv=False):

    # Reindex the scenarios to a particular time
    # horizon.
    time = pd.date_range(start="2022-05-01", end="2022-09-24", freq="d")

    #Assumes some social distancing / reducing number of partners to reduce contact numbers that affects R0 reduction 
    #This assumption needs to be calibrated better
    dates2= pd.date_range(start = social_distance_start_date,periods=30, freq='d')
    values2 = sigmoid_interpolation(start=1, end = social_distance_effect, num=len(dates2)) #contact rates reduced by 5%           

    #Basically it updates the values of R0 on the corresponding dates   
    social_distancing_dates_effects = dict(zip(dates2, values2))
    
    dataset = pd.read_csv(os.path.join(data_file))
    dataset = dataset[['diagnosis_date', 'count']]
    dataset = dataset.rename(columns={'count': 'cases'})
    dataset = dataset.rename(columns={'diagnosis_date': 'date'})
    dataset['importations'] = [0]*len(dataset)
    dataset['date'] = dataset['date'].apply(lambda x: pd.to_datetime(x))
    dataset = dataset.set_index('date').fillna(0)
    
    scenarios = dataset.reindex(time).fillna(0)
    
    #Initial importations
    scenarios["hypothetical"] = scenarios["importations"].copy()
    scenarios.loc["05-02-2022", "hypothetical"] = g_i
    
    
    # Set up the importation scenarios
    # Initialize model for hypothetical scenario
    hypothetical_scenario = LogNormalSEIR(S0=population,
                                        D_e=D_e,
                                        D_i=D_i,
                                        z_t=scenarios["hypothetical"].values,
                                        add_z_to_inf=True)                                                                                 
    hypothetical_scenario.sig_eps = monkeypox_sig_eps
    hypothetical_scenario.R0 = monkeypox_base_R0

    immune_num_t = pd.Series(0 * np.ones((len(time),)), index=time, name="immune_s1")
    immune_num_t = apply_intervention(immune_num_t , immune_num_effects)
    
    #Set up the attrack rate 
    beta_t = pd.Series(((monkeypox_base_R0 * (1/hypothetical_scenario.D_i))/hypothetical_scenario.S0) 
                    * np.ones((len(time),)),
                    index=time, name="beta_s1")
    
    # Do you want some social distancing? Modify attack rate based on function
    beta_t = apply_social_distancing_to_beta(beta_t, social_distancing_dates_effects)
    
    # Run replicates of scenarios
    population_samples = hypothetical_scenario.sample_scenario(beta_t.values, immune_num_t.values)

    # Save output
    append = "nyc_monkeypox_R0_%0.1f.csv" % (monkeypox_base_R0) 
    ts, ts_all = samples_2_df(population_samples, time=time)
    if write_output_to_csv:
        save_output_to_csv(ts, ts_all, append)
    return ts



if __name__ == "__main__":

    #Efficacy for monkeypox vaccinations after two doses (~85%); at the moment, most people are getting the first dose only 
    #Therefore assume that the efficacy is lower after the first dose only
    vacc_efficacy=0.728312295830168
    #Daily vaccination trends - THIS NEEDS TO BE UPDATED up to the current daily doses of vaccination then use the projections offerred by DoHMH as shown in the csv file
    vaccine_trend = pd.read_csv('doses-by-day.csv')
    vaccine= np.array(vaccine_trend['vx_doses'])
    
    #Defining different R0 values to test initially 
    R0_list = np.arange(1.5,3.0,0.1)#np.array([2.68]*10)#np.arange(2.6,2.8,0.01)
    error_list = []

    #Move people from S to R at 21 days after the first dose - assume that everyone will be eventually fully vaccinated
    vaccine_date = pd.date_range(start="2022-06-23", end="2021-08-31")
    vaccine_date_list = list(vaccine_date)
    immune_date_list = list(vaccine_date + timedelta(days=5))
    immune_date = [date.strftime('%Y-%m-%d') for date in immune_date_list]
    
    #Zip as a dictionary for vaccindation dates and the corresponding daily dose
    immune = dict(zip(immune_date, vaccine*vacc_efficacy))

    #Initial importations - these can be better calibrated and tested for sensitivity analyses
    global_importation_numbers = np.array([3])
       
    # Model parameters based on COVID-19 - need to be updated for monkeypox
    monkeypox_sig_eps = 0.7224738559305864
    monkeypox_rep_rate = 0.05 
    monkeypox_cfr = 0.0004 #case-fallity rate; as of August 20, 2022, WHO reported 11 deaths and 27814 cases
    #https://www.who.int/publications/m/item/multi-country-outbreak-of-monkeypox--external-situation-report--3---10-august-2022

    #Baseline exposed (i.e., incubation period) and infecious period
    D_e = 10 
    D_i = 21
    

    ts = sim_monkeypox(immune, **{'monkeypox_base_R0': 2.9986688954266496, 'g_i': 20, 'social_distance_effect': 0.6557988738343147, 'social_distance_start_date': "2022-07-27"})
    #**{'monkeypox_base_R0': 4.250724804162295, 'g_i': 18, 'social_distance_effect': 0.11352545356229893})
    cost = calc_cost(ts)
    plot_results(ts, percent_diagnosed = 0.8989031388159953 )
#                                    save_output_to_csv(hypothetical_scenario, population_samples, time=time, append=append, folder = scenario_char, plot=False)
