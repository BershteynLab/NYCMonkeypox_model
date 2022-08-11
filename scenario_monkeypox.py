
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

# For time-series modeling
from seir_monkeypox import LogNormalSEIR
from helpers_monkeypox import *
np.set_printoptions(threshold=np.inf)

#sys.path.append(os.pardir)

if __name__ == "__main__":

    output_path = "/Users/haeyoungkim/Dropbox (NYU Langone Health)/NYU postdoc/MonkeyPox/"    
    data_file = 'cases-by-day.csv'    

    # Set up the initial susceptible population by getting the population of MSM being simulated
    population = 242000
    region = 'nyc'

    #Efficacy for monkeypox vaccinations after two doses (~85%); at the moment, most people are getting the first dose only 
    #Therefore assume that the efficacy is lower after the first dose only
    vacc_efficacy=0.55
    
    #Daily vaccination trends - THIS NEEDS TO BE UPDATED up to the current daily doses of vaccination then use the projections offerred by DoHMH as shown in the csv file
    vaccine_trend = pd.read_csv('monkeypox_vx_trend.csv')
    vaccine= np.array(vaccine_trend['vx_doses'])
    
    #Defining different R0 values to test initially 
    R0_list = np.array([2.2])
    

    #Move people from S to R at 21 days after the first dose - assume that everyone will be eventually fully vaccinated
    vaccine_date_list = list(pd.date_range(start="2022-06-23", end="2021-08-31"))
    immune_date_list = list(pd.date_range(start="2022-07-14",  end="2022-09-22"))
    immune_date = [date.strftime('%Y-%m-%d') for date in immune_date_list]
    
    #Zip as a dictionary for vaccindation dates and the corresponding daily dose
    immune = dict(zip(immune_date, vaccine*vacc_efficacy))

    #Initial importations - these can be better calibrated and tested for sensitivity analyses
    global_importation_numbers = np.array([3])
       
    # Model parameters based on COVID-19 - need to be updated for monkeypox
    monkeypox_sig_eps = 0.7224738559305864
    monkeypox_rep_rate = 0.05 
    monkeypox_cfr = 0.016 #case-fallity rate

    #Baseline exposed (i.e., incubation period) and infecious period
    D_e = 10 
    D_i = 14
    
    #Create output csv files 
    def save_output_to_csv(model, samples, time, append='', folder=True, plot=True):
            
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
            
            # Save it to your directory
            output_dir = os.path.join(output_path,"simulations/")
            os.makedirs(os.path.join(output_path,"simulations/"), exist_ok=True)
                
            output_newinfections.to_csv(os.path.join(output_dir, "nyc_new_infections_%s" % append)) #Number of new daily infections only 
            output_combined.to_csv(os.path.join(output_dir, "nyc_combined_%s" % append)) #Number for all compartments


    for R0_value in range(len(R0_list)): 
        
        monkeypox_base_R0 = R0_list[R0_value]
        monekypox_base_R0_str = str(monkeypox_base_R0)
        

        for g_i in global_importation_numbers:
             
            # Save output?
            save_output = True 

            #Assumes some social distancing / reducing number of partners to reduce contact numbers that affects R0 reduction 
            #This assumption needs to be calibrated better
            dates2= ["2022-06-05"]
            values2 = [monkeypox_base_R0 * 0.95] #contact rates reduced by 5%           

            #Basically it updates the values of R0 on the corresponding dates   
            social_distancing_dates_effects = dict(zip(dates2, values2))

            os.path.join(output_path, "Re_estimate/")
            os.makedirs(os.path.join(output_path,"Re_estimate/"), exist_ok=True)
                
            with open(os.path.join(output_path, "Re_estimate/")+ "Rt_estimate_R0_value" + monekypox_base_R0_str + ".csv", "w") as f: 
               for key in social_distancing_dates_effects.keys():
                   f.write("%s,%s\n"%(key,social_distancing_dates_effects[key]))

            #Number of people who moves from S to R via vaccination 
            immune_num_effects= immune

            # Reindex the scenarios to a particular time
            # horizon.
            time = pd.date_range(start="2022-05-01", end="2022-08-31", freq="d")
            
            dataset = pd.read_csv(os.path.join(data_file))
            dataset = dataset[['diagnosis_date', 'count']]
            dataset = dataset.rename(columns={'count': 'cases'})
            dataset = dataset.rename(columns={'diagnosis_date': 'date'})
            dataset['importations'] = [0]*len(dataset)
            dataset['importations'].iloc[0] = 1
            dataset['date'] = dataset['date'].apply(lambda x: pd.to_datetime(x))
            dataset = dataset.set_index('date').fillna(0)
            
            scenarios = dataset.reindex(time).fillna(0)
            
            #Initial importations
            scenarios["hypothetical"] = scenarios["importations"].copy()
            scenarios.loc["05-10-2022", "hypothetical"] = g_i
            print(scenarios.loc[scenarios["hypothetical"] != 0])
            
            
            # Set up the importation scenarios
            # Initialize model for hypothetical scenario
            hypothetical_scenario = LogNormalSEIR(S0=population,
                                                  D_e=D_e,
                                                  D_i=D_i,
                                                  z_t=scenarios["hypothetical"].values)                                                                                 
            hypothetical_scenario.sig_eps = monkeypox_sig_eps
            hypothetical_scenario.rep_rate = monkeypox_rep_rate
            hypothetical_scenario.R0 = monkeypox_base_R0
 
            immune_num_effects= immune
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
            if save_output:
                append = "nyc_monkeypox_R0_%0.1f.csv" % (monkeypox_base_R0) 
                save_output_to_csv(hypothetical_scenario, population_samples, time=time, append=append, plot=False)
#                                    save_output_to_csv(hypothetical_scenario, population_samples, time=time, append=append, folder = scenario_char, plot=False)
