import PAINT_analysis_functions as Func
import os
import numpy as np
from tqdm import tqdm

color = ["green", "red"]
sample_list = ["H1_", "H24_", "H72_", "H168_"]

# time = time_range[0]
# sample = "0%N3_"
drift_correct = True
fiducial_filter = False
select_roi = True
mult_roi = True
frame_rate = 10  # in Hz

initial_time = np.arange(1, 20*60*frame_rate, 5*60*frame_rate)
final_time = initial_time + 5*60*frame_rate - 1
time_range = ["_" + str(initial_time[i]) + "-" + str(final_time[i]) for i in range(initial_time.size)]

for sample in sample_list:
    for time in time_range:
        name_drift = sample + color[0] + time
        name = sample + color[1] + time
        path = name.split("_")[0] + "_" + name.split("_")[-1]

        if not os.path.exists(path):
            os.mkdir(path)

        # Drift correction module
        if drift_correct:
            if not os.path.exists(path + "/DriftCorrection/x_drift.npy"):
                Func.DriftCorrection(name_drift, path).drift_correct(fiducial_filter)

        # Raw data plotting prior to any filtering or analysis
        Func.PAINT(name, path).plot_coord_raw()
        Func.PAINT(name, path).plot_prop_raw()

        # Pre-processing of data - data is drift-corrected, and filtered according to their sigma, intensity, offset and
        # uncertainty values
        Func.PAINTFiltered(name, path).pre_processing(drift_correct, filter_intensity=False, plt_save=True)

        if time == time_range[0]:
            # Clustering algorithm
            Func.Clustering(path).run(select_roi=select_roi, multiple_roi=mult_roi)

            # Lifetime analysis
            Func.LifetimeAnalysis(path, frame_rate).lifetime(multiple_roi=mult_roi)
            # Func.LifetimeAnalysis(path, frame_rate).compute_traces_per_clust(219)

            # Determine molecule density
            Func.MoleculeDensity(path).run(multiple_roi=mult_roi, choose_roi=select_roi)

            # Determine molecular distribution of binders
            Func.ClarkEvansTest(path).run(multiple_roi=mult_roi, choose_roi=select_roi)
