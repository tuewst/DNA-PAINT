import PAINT_analysis_functions
import PAINT_analysis_functions as Func
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def iterate_for_each_block(samples_, time_ranges, select_roi, mult_roi, framerate, drift):
    for blocks in tqdm(np.arange(2, len(time_ranges) + 1, 1)):
        for sample_ in samples_:
            folder_lists = [sample_.split("_")[0] + time for time in time_ranges]
            path_ = sample_ + str(blocks) + "blocks"
            total_frame = frame_per_block * blocks

            Func.MergedLoc(folder_lists, path_, blocks, frame_per_block).merge_data(drift, total_frame)

            # Clustering algorithm
            Func.Clustering(path_).run(select_roi=select_roi, multiple_roi=mult_roi)

            # Lifetime analysis
            Func.LifetimeAnalysis(path_, framerate).lifetime(multiple_roi=mult_roi)

            Func.IntensityCheck(path_, framerate).run(multiple_roi=mult_roi)

            # Determine molecule density
            Func.MoleculeDensity(path_).run(multiple_roi=mult_roi, choose_roi=select_roi)

            # Determine molecular distribution of binders
            Func.ClarkEvansTest(path_).run(multiple_roi=mult_roi, choose_roi=select_roi)


sample_list = samples = ["1%N3noncomp-25pM_"]


# sample_list = samples = ["1%N3H168-2_", "1%N3H168-3_", "10%N3H168-2_", "10%N3H168-3_",
#                          "10%N3noncompH168-2_", "10%N3noncompH168-3_"]

# sample_list = samples = ["1%N3noncomp-25pM_", "10%N3noncomp-25pM_"]

# sample_list = ["0.01mgml_", "0.025mgml_", "0.05mgml_", "0.5mgml_"]

# sample_list = ["50pM_", "100pM_", "250pM_", "350pM_"]

# sample_list = ["0%N3cort_", "10%N3cort_"]

# sample_list = ["0%N30.1_", "0%N30.5_", "0%N31_", "1%BSA_", "1%N3ctrl_", "10%N3ctrl_"]
# sample_list = ["1%N3noncomp_", "10%N3noncomp_", "20%N3noncomp_"]

# sample_list = ["10%N3mismatchH1_", "10%N3mismatchH24_", "10%N3mismatchH72_", "10%N3mismatchH168_"]  # "10%N3mismatchH1_", "10%N3mismatchH24_", "10%N3mismatchH72_", "10%N3mismatchH168_"
# sample_list = ["1%N3H1_", "1%N3H24_", "1%N3H72_", "1%N3H168_"]
# sample_list = ["H1_", "H24_", "H72_", "H168_"]


n_blocks = 4
drift_correct = True
roi = True
frame_rate = 10  # in Hz
frame_per_block = 5*60*frame_rate

initial_time = np.arange(1, 20*60*frame_rate, frame_per_block)
final_time = initial_time + frame_per_block - 1
time_range = ["_" + str(initial_time[i]) + "-" + str(final_time[i]) for i in range(initial_time.size)]

# Run for each block; Comment when not used
# iterate_for_each_block(sample_list, time_range, roi, roi, frame_rate, drift_correct)

# Run for specific block; Comment when not used
for sample in sample_list:
    folder_list = [sample.split("_")[0] + time for time in time_range]
    path = sample + str(n_blocks) + "blocks"
    total_frames = frame_per_block * n_blocks

    # Func.MergedLoc(folder_list, path, n_blocks, frame_per_block).merge_data(drift_correct, total_frames)

    # # Clustering algorithm
    # Func.Clustering(path).run(select_roi=roi, multiple_roi=roi)

    # # Lifetime analysis
    # Func.LifetimeAnalysis(path, frame_rate).lifetime(multiple_roi=roi)
    # Func.LifetimeAnalysis(path, frame_rate).compute_traces_per_clust(219)

    # Check for intensity, compute unfiltered LT
    # Func.IntensityCheck(path, frame_rate).run(multiple_roi=roi)
    Func.IntensityCheck(path, frame_rate).plot_intensity(plt_save=False, inv_thresh=2.5e-8, multiple_roi=False)
    # Func.IntensityCheck(path, frame_rate).classify_intensity_in_cluster()

    # # Determine molecule density
    # Func.MoleculeDensity(path).run(multiple_roi=roi, choose_roi=roi)
    #
    # # Determine molecular distribution of binders
    # Func.ClarkEvansTest(path).run(multiple_roi=roi, choose_roi=roi)
