import os

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Helper functions
def axes_setup(ax):

    ax.spines["left"].set_position(("axes", -0.025))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return None

def low_mid_high(data_samples, low, high):

    low_bound = np.percentile(data_samples, low, axis=0)
    high_bound = np.percentile(data_samples, high, axis=0)

    return low_bound, high_bound


def plotting_scenario(ax, samples, data, model, time):

    # opacity associated with bounds
    alpha_bounds = {0.1: [1, 99], 0.3: [2.5, 97.5], 0.8: [25, 75]}

    label = "Monkeypox model, " + "{0:0.0f}".format(
        100 * model.rep_rate) + r"% case detection rate, $\bf{multiple}$ $\bf{early}$ $\bf{importations}$"

    for i, a in enumerate(alpha_bounds):
        l, h = low_mid_high(samples, alpha_bounds[a][0], alpha_bounds[a][1])
        if a == max(alpha_bounds):
            ax.fill_between(time, l, h, color="#F98866", alpha=a, label=label)
            # Print out a the low and high values
            cases_in_model = pd.DataFrame(np.array([l, h]).T,
                                          index=time,
                                          columns=["low", "high"])
            print("\nPrediction intervals for a few days...")
        else:
            ax.fill_between(time, l, h, color="#F98866", alpha=a)

    ax.xaxis.set_major_locator(mdates.MonthLocator())  # to get a tick every 15 days
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plotting_cases_importations(ax, data)

    return None

def exp_function(x, a, b):
    phase = a*(np.exp(b*x))
    return (phase)

def sigmoid(x, a, b, c, d):
    y = a*(np.exp(b*(x-c))/(np.exp(b*(x-c))+1))+d
    return (y)

def plotting_cases_importations(ax, data, annotate_importations=True):
    # Plot the data
    ax.plot(data["cases"],  # /model.p,
            ls="None", color="k", marker="o", markersize=10, label="Data from NYC")

    # And importations
    ax.plot(data.loc[data["importations"] != 0]["importations"], ls="None", marker="o", markersize=12,
            markeredgecolor="xkcd:red wine", markerfacecolor="None", markeredgewidth=2, label="Importations")

    # Add annotations for Wuhan and Italy
    axis_to_data = ax.transAxes + ax.transData.inverted()

    if annotate_importations:
        ax.axvline(pd.to_datetime("2022-05-10"), ymin=0.08, ymax=0.3, ls="dashed", color="k")
        ax.text(pd.to_datetime("2022-05-10"), axis_to_data.transform((0, 0.4))[1], "Global", rotation=90,
                verticalalignment="center", horizontalalignment="center", color="k", fontsize=24)

    # Finish up
    ax.legend(frameon=False, loc=2)
    ax.set_ylabel("Daily detections")

def save_output_to_csv(model, samples, time, append='', folder=True, plot=True):
     # output for new infections
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
    
    output_combined = pd.concat([outputS, outputE, outputI, outputR, output_vacc_actual], output_newinfections, axis=1).reindex(outputS.index)
    
     
    # Save it to your directory
    output_dir = os.path.join(output_path, "simulations/")
    os.makedirs(os.path.join(output_path, "simulations/"), exist_ok=True)
        
    output_newinfections.to_csv(os.path.join(output_dir, "nyc_new_infections_%s" % append)) #Number of new daily infections only 
    output_combined.to_csv(os.path.join(output_dir, "nyc_combined_%s" % append)) #Number for all compartments

#    
    if plot:
        # Make a plot
        figure, ax = plt.subplots(figsize=(18, 7))
        axes_setup(ax)
        ax.plot(output.mean(axis=1), c="grey", label="output 1")
        ax.plot(output2.mean(axis=1), c="k", label="output 2")
        # ax.axhline(population,ls="dashed",color="k")
        ax.legend()
        figure.tight_layout()

        # Finish up
        plt.show()

def apply_social_distancing_to_beta(beta, sd_effects_dates):
    effects = list(sd_effects_dates.values())
    
    for i, d in enumerate(sd_effects_dates):
        beta.loc[d:] = effects[i] * beta.values[0]
        
    return beta

def apply_intervention(theta, intervention_effects_dates):
    effects_theta = list(intervention_effects_dates.values())
    
    for i, d in enumerate(intervention_effects_dates):
        theta.loc[d:] = effects_theta[i]
        
    return theta


def prevalence_and_mortality(prevalence_date, model, time, population_samples, cfr):

    # Report on prevalence
    date = pd.to_datetime(prevalence_date)
    step = np.argmin([np.abs(t - date) for t in time])
    h_i_samples = population_samples[:, 1, step] - \
                  (1. - (1. / model.D_e)) * population_samples[:, 1, step - 1]
    h_mean, h_low, h_high = np.percentile(h_i_samples, 50), np.percentile(h_i_samples, 25), \
                            np.percentile(h_i_samples, 75)

    # Report on mortality
    mortality_date = date - pd.to_timedelta(25, unit="d")
    m_step = np.argmin([np.abs(t - mortality_date) for t in time])
    h_cumulative_inf = model.S0 - population_samples[:, 0, :]
    m_samples = cfr * h_cumulative_inf[:, m_step]
    m_mean, m_low, m_high = np.percentile(m_samples, 50), np.percentile(m_samples, 2.5), \
                            np.percentile(m_samples, 97.5)

    # Print outputs on active infections and mortality
    print("\nOn {}...".format(date.strftime("%m/%d/%Y")))
    print("Hypothetical active infections = {0:0.0f} ({1:0.0f}, {2:0.0f})".format(h_mean, h_low, h_high))
    print(mortality_date)
    print("Hypothetical mortality = {0:0.0f} ({1:0.0f}, {2:0.0f})".format(m_mean, m_low, m_high))
    print("Hypothetical cum infections = %i" % h_cumulative_inf.mean(axis=0)[-1])
