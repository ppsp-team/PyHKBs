import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from scipy import stats


with open(r"multi_agent_simulation_runs_flavour_0.pickle", "rb") as input_file:
    average_final_distance_0 = pickle.load(input_file)

with open(r"multi_agent_simulation_runs_flavour_1.pickle", "rb") as input_file:
    average_final_distance_1 = pickle.load(input_file)

with open(r"multi_agent_simulation_runs_flavour_2.pickle", "rb") as input_file:
    average_final_distance_2 = pickle.load(input_file)

metric = ['Average distance to option', 'Average distance to option', 'Average inter-agent distance']

for i in range(3):
    print(stats.kruskal(average_final_distance_0[i, :], average_final_distance_1[i, :], average_final_distance_2[i, :]))
    print(stats.mannwhitneyu(average_final_distance_0[i, :], average_final_distance_1[i, :]))
    print(stats.mannwhitneyu(average_final_distance_1[i, :], average_final_distance_2[i, :]))
    print(stats.mannwhitneyu(average_final_distance_0[i, :], average_final_distance_2[i, :]))




    df = pd.melt( pd.DataFrame( {"asocial":average_final_distance_0[i, :], "socially sensitive":average_final_distance_1[i, :], "stimulus emitting": average_final_distance_2[i, :] }), var_name = 'Agent flavour', value_name = metric[i]) 
    sns.swarmplot(data = df, x = 'Agent flavour', y = metric[i], hue = 'Agent flavour')
    plt.show()
    

