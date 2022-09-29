import PAINT_plot_functions as Func
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

sample_list = ["0%N3_", "1%N3_", "10%N3_", "10%N3ctrl_", "10%N3100pM_"]
param_list = ["tau_bright", "tau_dark", "density", "z_scores", "density_filter_repeatedframe", "z_scores_filter_repeatedframe"]
all_params = False
param = param_list[0]
Func.PlotInitialize(10, "", sample_list, single_merge_compare=1, n_blocks=4, all_params=all_params, param=param).run()

# tau_bright = []
# cdf_bright = []
# k = 0
# colors = ["b", "g", "r", "c", "m"]
# for path in paths:
#     tau = Func.get_all_data(path, combine_roi=True)["tau_bright"]
#     tau, f, pred = Func.obtain_pred_for_lifetimes(tau, 10, 2)
#     plt.scatter(tau, 1 - f, lw=2, label=samples[k], color=colors[k], s=1)
#     plt.plot(tau, Func.exp_lifetimes2(tau, pred[0][0], pred[0][1], pred[0][2], pred[0][3]), "--", color=colors[k])
#     sigma_popt = np.sqrt(np.diag(pred[1]))
#     bound_upper = Func.exp_lifetimes2(tau, *pred[0] - sigma_popt)
#     bound_lower = Func.exp_lifetimes2(tau, *pred[0] + sigma_popt)
#     plt.fill_between(tau, bound_lower, bound_upper, color=colors[k], alpha=0.15)
#     tau_bright.append(tau)
#     cdf_bright.append(1-f)
#     k += 1
# plt.yscale('log')
# plt.legend(fancybox=True, loc='upper right')
# plt.show()
# print(st.ks_2samp(tau_bright[1], tau_bright[0]))
