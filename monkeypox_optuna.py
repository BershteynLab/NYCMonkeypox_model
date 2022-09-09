import optuna
import scenario_monkeypox
import datetime as dt
import pandas as pd
import numpy as np

#Daily vaccination trends - THIS NEEDS TO BE UPDATED up to the current daily doses of vaccination then use the projections offerred by DoHMH as shown in the csv file
vaccine_trend = pd.read_csv('monkeypox_vx_trend_8-19.csv', parse_dates=["date"])
vaccine= np.array(vaccine_trend['vx_doses'])

#Move people from S to R at 21 days after the first dose - assume that everyone will be eventually fully vaccinated
vaccine_date = pd.date_range(start="2022-06-23", end="2021-09-25")
vaccine_date_list = list(vaccine_date)
immune_date_list = list(vaccine_date + dt.timedelta(days=5))
immune_date = [date.strftime('%Y-%m-%d') for date in immune_date_list]


def objective(trial : optuna.trial.BaseTrial):
    """
    Objective function for optuna. First, the parameters are set. Then, the simulation is run.
    Finally, the mean squared error is calculated and returned.
    """
    monkeypox_base_R0 = trial.suggest_float('monkeypox_base_R0', 0.5, 4)
    g_i = trial.suggest_int("g_i",1, 20)
    social_distance_effect = trial.suggest_float('social_distance_effect', 0.5, 1. )
    percent_diagnosed = trial.suggest_float("percent_diagnosed", 0.05, 0.5)
    social_distance_date_offset = trial.suggest_int("social_distance_date_offset", -15, 15)
    vaccine_efficacy = trial.suggest_float("vaccine_efficacy", 0.55, 0.95)
    incubation_period = trial.suggest_float("incubation_period", low=6.6, high=10.9) #https://doi.org/10.2807/1560-7917.ES.2022.27.24.2200448
    infectious_period = trial.suggest_float("infectious_period", low=14, high=28)
    population = 70_180 #trial.suggest_float("population", low=5_000, high=200_000)
    ts = ts_wrapper(vaccine_efficacy,infectious_period=infectious_period,
                    incubation_period=incubation_period, 
                    monkeypox_base_R0=monkeypox_base_R0, g_i=g_i, 
                    social_distance_effect=social_distance_effect, 
                    social_distance_date_offset=social_distance_date_offset, population=population)
    return scenario_monkeypox.calc_cost(ts, percent_diagnosed)

def ts_wrapper(vaccine_efficacy, infectious_period, incubation_period, monkeypox_base_R0, g_i, social_distance_effect, social_distance_date_offset, population=70_180, **kwargs):
    """
    Wrapper around the simulation for easy calling by the objective function.
    """
    social_distance_date  = dt.date(2022, 7, 15) + dt.timedelta(days=social_distance_date_offset)
    immune = dict(zip(immune_date, vaccine*vaccine_efficacy))
    ts = scenario_monkeypox.sim_monkeypox(immune, monkeypox_base_R0=monkeypox_base_R0, g_i=g_i, 
                                            social_distance_effect=social_distance_effect, social_distance_start_date=social_distance_date, 
                                            D_e = incubation_period, D_i=infectious_period, population=population, **kwargs)
    return ts


# Saved params for easy access
params_low_pop = {'monkeypox_base_R0': 3.925127737879024, 'g_i': 5, 'social_distance_effect': 0.500790748619936, 'social_distance_date_offset': 0, 'vaccine_efficacy': 0.936377094886125, 'incubation_period': 9.13966250883466, 'infectious_period': 19.915887897884716, 'population': 10815.236226783223}
params_const_pop = {'monkeypox_base_R0': 3.780016395948963, 'g_i': 19, 'social_distance_effect': 0.5640281290816059, 'percent_diagnosed': 0.06019924509957085, 'social_distance_date_offset': 1, 'vaccine_efficacy': 0.7729386265600666, 'incubation_period': 7.498379751088684, 'infectious_period': 14.015280924828911}


def best_fit(params: dict):
    """
    Plot a set of params and save the results to a csv
    """
    percent_diagnosed = params.pop('percent_diagnosed', 1)
    ts = ts_wrapper(**params, write_output_to_csv=True)
    scenario_monkeypox.plot_results(ts, percent_diagnosed=percent_diagnosed)
    (ts * percent_diagnosed).to_csv("simulated_results.csv")


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=1_000)
    print(study.best_params)
    ts = best_fit( study.best_params )

