import PAINT_plot_functions as Func
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import scipy

# sample_list = ["H1_", "H24_", "H72_", "H168_", "1%N3H1_", "1%N3H24_", "1%N3H72_", "1%N3H168_",
#                "10%N3mismatchH1_", "10%N3mismatchH24_", "10%N3mismatchH72_", "10%N3mismatchH168_"]
# sample_list = ["H1_", "H24_", "H72_", "H168_",
#                "10%N3H1-2_", "10%N3H24-2_", "10%N3H72-2_", "10%N3H168-2_",
#                "10%N3H1-3_", "10%N3H24-3_", "10%N3_", "10%N3H168-3_"]
sample_list = ["1%N3H1_", "1%N3H24_", "1%N3H72_", "1%N3H168_",
               "1%N3H1-2_", "1%N3H24-2_", "1%N3H72-2_", "1%N3H168-2_",
               "1%N3H1-3_", "1%N3H24-3_", "1%N3H72-3_", "1%N3H168-3_"]
# sample_list = ["10%N3mismatchH1_", "10%N3mismatchH24_", "10%N3mismatchH72_", "10%N3mismatchH168_",
#                "10%N3noncompH1-2_", "10%N3noncompH24-2_", "10%N3noncompH72-2_", "10%N3noncompH168-2_",
#                "10%N3noncompH1-3_", "10%N3noncompH24-3_", "10%N3noncompH72-3_", "10%N3noncompH168-3_"]
# sample_list = ["H1_", "H24_", "H72_", "H168_", "1%N3H1_", "1%N3H24_", "1%N3H72_", "1%N3H168_",
#                "10%N3mismatchH1_", "10%N3mismatchH24_", "10%N3mismatchH72_", "10%N3mismatchH168_"]
# sample_list = samples = ["1%N310mer-25pM_", "1%N3H72_", "1%N3H72-2_", "1%N3H72-3_"]
framerate = 10
# imager_conc = 250 * 10 ** -12
paths = Func.PlotInitialize(10, "", sample_list, single_merge_compare=1, n_blocks=4, total_time_min=20,
                            time_interval_min=5).obtain_path()
# Func.get_tau_info(sample_list, paths, framerate, 2, 2)
Func.get_density_z_info(sample_list, paths, "unfiltered")
# print(Func.read_pkl_info("density_z_info"))
# print(Func.read_pkl_info())
# Func.plot_tau()
Func.plot_density_z(save=True, save_name="_1%N3")
# Func.plot_density_bar_plot()
