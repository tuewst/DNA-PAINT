import PAINT_plot_functions as Func
import numpy as np

# sample_list = samples = ["10%N310mer-25pM_", "10%N310mer-50pM_"]
sample_list = samples = ["1%N310mer-25pM_", "1%N3noncomp-25pM_"]
# sample_list = samples = ["10%N310mer-50pM_", "H72_", "10%N3_", "10%N32_", "0.5mgml_"]
# sample_list = samples = ["1%N3H72_", "250pM_", "1%N3_", "1%N32_"]
# sample_list = samples = ["H72_", "10%N3_", "10%N32_", "0.5mgml_"]
# sample_list = samples = ["10%N3mismatchH72_", "10%N3noncomp_"]

# sample_list = samples = ["1LP15%_", "1LP35%_", "1LP55%_"]

# sample_list = samples = ["1%N310mer-50pM_", "10%N310mer-50pM_"]
# sample_list = samples = ["1%N310mer-25pM_", "10%N310mer-25pM_"]

# sample_list = samples = ["0.01mgml_", "0.025mgml_", "0.05mgml_", "0.5mgml_"]

# sample_list = samples = ["100pM_"]  # , "100pM_", "250pM_", "350pM_"

# sample_list = samples = ["1%N3_", "10%N3_", "15%N3_", "20%N3_", "20%N3ctrl_"]  # , "20%N3ctrl_"

# sample_list = samples = ["10%N3mismatchH1_", "10%N3mismatchH24_", "10%N3mismatchH72_", "10%N3mismatchH168_"]
# sample_list = samples = ["H1_", "H24_", "H72_", "H168_", "10%N3noncomp_"]  #, "10%N3noncomp_"
# sample_list = samples = ["1%N3noncomp_", "1%N3H1_", "1%N3H24_", "1%N3H72_", "1%N3H168_"]  #, "1%N3noncomp_"
# sample_list = samples = ["1%N3H1-2_", "1%N3H1-3_", "1%N3H24-2_", "1%N3H24-3_", "1%N3H72-2_", "1%N3H72-3_",
#                          "1%N3H168-2_", "1%N3H168-3_"]
# sample_list = samples = ["10%N3H1-2_", "10%N3H1-3_", "10%N3H24-2_", "10%N3H24-3_", "10%N3H72-2_", "10%N3H168-2_",
#                          "10%N3H168-3_"]
# sample_list = samples = ["10%N3noncompH1-2_", "10%N3noncompH1-3_", "10%N3noncompH24-2_", "10%N3noncompH24-3_"
#                          , "10%N3noncompH72-2_", "10%N3noncompH72-3_", "10%N3noncompH168-2_", "10%N3noncompH168-3_"]

# sample_list = samples = ["0%N30.5_", "1%N3ctrl_", "1%N3noncomp_"]
# sample_list = samples = ["0%N30.1_", "0%N30.5_", "0%N31_", "1%BSA_", "1%N3ctrl_", "10%N3ctrl_", "1%N3noncomp_", "10%N3noncomp_", "20%N3noncomp_"]
# sample_list = samples = ["1%N3noncomp_", "10%N3noncomp_", "20%N3noncomp_"]

# x_list = [np.append(np.array([-5, 1, 10]), np.arange(20, 101, 20)), np.arange(5, 21, 5)]
# x_label_name = ["PLL-g-PEG-N3/Total PLL (%)", "Measurement duration (min)"]
# save_name_list = ["", ""]
# x_list = [[1, 24, 72, 168, "-ve control"], np.arange(5, 21, 5)]
x_list = [None, np.arange(5, 21, 5)]
save_name_list = ["_10%N3", ""]
save = False
x_label_name = ["Total [PLL] (mg/mL)", "Measurement duration (min)"]

frame_rate = 10  # in Hz
total_time = 20
time_interval = 5
single_merge = True
single_merge_block = 1  # single=0, merge=1, compare_block=2
n_blocks = 4

param_list = ["tau_bright_unfiltered", "tau_dark_unfiltered", "density", "z_scores",
              "density_filter_repeatedframe", "z_scores_filter_repeatedframe", "density_unfiltered",
              "z_scores_unfiltered"]  # "tau_dark" "tau_bright"
all_params = False
param = param_list[0]

if all_params:
    if single_merge:
        if any(isinstance(i, list) for i in samples):
            for i in range(len(samples)):
                Func.PlotInitialize(frame_rate, save_name_list[i], samples[i], x_list=x_list[i],
                                    x_label_name=x_label_name[i], single_merge_compare=single_merge_block,
                                    n_blocks=n_blocks, total_time_min=total_time, time_interval_min=time_interval).run()
        else:
            Func.PlotInitialize(frame_rate, save_name_list[0], samples, x_list=x_list[0], x_label_name=x_label_name[0],
                                single_merge_compare=single_merge_block, n_blocks=n_blocks, total_time_min=total_time,
                                time_interval_min=time_interval).run()
    else:
        for sample in sample_list:
            Func.PlotInitialize(frame_rate, save_name_list[-1], sample, x_list=x_list[-1], x_label_name=x_label_name[-1],
                                single_merge_compare=2, total_time_min=total_time, time_interval_min=time_interval).run()
else:
    if single_merge:
        if any(isinstance(i, list) for i in samples):
            for i in range(len(samples)):
                Func.PlotInitialize(frame_rate, save_name_list[i], samples[i], x_list=x_list[i],
                                    x_label_name=x_label_name[i], single_merge_compare=single_merge_block,
                                    n_blocks=n_blocks, all_params=all_params, param=param, total_time_min=total_time,
                                    time_interval_min=time_interval).run()
        else:
            Func.PlotInitialize(frame_rate, save_name_list[0], samples, x_list=x_list[0], x_label_name=x_label_name[0],
                                single_merge_compare=single_merge_block, n_blocks=n_blocks, all_params=all_params,
                                param=param, total_time_min=total_time, time_interval_min=time_interval).run(save=save)
    else:
        for sample in sample_list:
            Func.PlotInitialize(frame_rate, save_name_list[-1], sample, x_list=x_list[-1], x_label_name=x_label_name[-1],
                                single_merge_compare=2, all_params=all_params, param=param, total_time_min=total_time,
                                time_interval_min=time_interval).run()

paths = Func.PlotInitialize(frame_rate, "", sample_list, single_merge_compare=1, n_blocks=4, total_time_min=total_time,
                            time_interval_min=time_interval).obtain_path()
# Func.get_tau_info(sample_list, paths, frame_rate)
Func.get_clust_info(sample_list, paths)
# Func.get_density_z_info(sample_list, paths)
# print(Func.read_pkl_info())
print(Func.read_pkl_info("clust_info"))
# Func.plot_tau(True)
