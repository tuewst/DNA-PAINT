import PAINT_analysis_functions as Func
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


color = ["green", "red"]
sample_list = ["H1_", "H24_", "H72_", "H168_"]

# sample = "1%N3_"
# n_blocks = 4
drift_correct = True
select_roi = True
mult_roi = True
frame_rate = 10  # in Hz
frame_per_block = 5*60*frame_rate

initial_time = np.arange(1, 20*60*frame_rate, 5*60*frame_rate)
final_time = initial_time + 5*60*frame_rate - 1
time_range = ["_" + str(initial_time[i]) + "-" + str(final_time[i]) for i in range(initial_time.size)]

for n_blocks in np.arange(2, 5, 1):
    for sample in tqdm(sample_list):
        folder_list = [sample.split("_")[0] + time for time in time_range]
        path = sample + str(n_blocks) + "blocks"
        total_frames = frame_per_block * n_blocks

        Func.MergedLoc(folder_list, path, n_blocks).merge_data(drift_correct, total_frames)

        # Clustering algorithm
        Func.Clustering(path).run(select_roi=select_roi, multiple_roi=mult_roi)

        # Lifetime analysis
        Func.LifetimeAnalysis(path, frame_rate).lifetime(multiple_roi=mult_roi)
        # Func.LifetimeAnalysis(path, frame_rate).compute_traces_per_clust(219)

        # Determine molecule density
        Func.MoleculeDensity(path).run(multiple_roi=mult_roi, choose_roi=select_roi)

        # Determine molecular distribution of binders
        Func.ClarkEvansTest(path).run(multiple_roi=mult_roi, choose_roi=select_roi)

# Autocorrelation analysis from lbFCS
# Func.Autocorrelation(path).run()
