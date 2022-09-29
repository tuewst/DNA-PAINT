import PAINT_plot_functions as Func
import numpy as np
import matplotlib.pyplot as plt

sample_list = ["H1_", "H24_", "H72_", "H168_"]
samples = sample_list
x_list = [[1, 1 * 24, 3 * 24, 7 * 24], np.arange(5, 21, 5)]
x_label_name = ["SPAAC reaction duration (hr)", "Measurement duration (min)"]
save_name_list = ["", ""]

frame_rate = 10  # in Hz
single_merge = True
single_merge_block = 1  # single=0, merge=1, compare_block=2
n_blocks = 4

param_list = ["tau_bright", "tau_dark", "density", "z_scores", "density_filter_repeatedframe", "z_scores_filter_repeatedframe"]
all_params = False
param = param_list[0]

if all_params:
    if single_merge:
        if any(isinstance(i, list) for i in samples):
            for i in range(len(samples)):
                Func.PlotInitialize(frame_rate, save_name_list[i], samples[i], x_list=x_list[i],
                                    x_label_name=x_label_name[i], single_merge_compare=single_merge_block,
                                    n_blocks=n_blocks).run()
        else:
            Func.PlotInitialize(frame_rate, save_name_list[0], samples, x_list=x_list[0], x_label_name=x_label_name[0],
                                single_merge_compare=single_merge_block, n_blocks=n_blocks).run()
    else:
        for sample in sample_list:
            Func.PlotInitialize(frame_rate, save_name_list[-1], sample, x_list=x_list[-1], x_label_name=x_label_name[-1],
                                single_merge_compare=2).run()
else:
    if single_merge:
        if any(isinstance(i, list) for i in samples):
            for i in range(len(samples)):
                Func.PlotInitialize(frame_rate, save_name_list[i], samples[i], x_list=x_list[i],
                                    x_label_name=x_label_name[i], single_merge_compare=single_merge_block,
                                    n_blocks=n_blocks, all_params=all_params, param=param).run()
        else:
            Func.PlotInitialize(frame_rate, save_name_list[0], samples, x_list=x_list[0], x_label_name=x_label_name[0],
                                single_merge_compare=single_merge_block, n_blocks=n_blocks, all_params=all_params,
                                param=param).run()
    else:
        for sample in sample_list:
            Func.PlotInitialize(frame_rate, save_name_list[-1], sample, x_list=x_list[-1], x_label_name=x_label_name[-1],
                                single_merge_compare=2, all_params=all_params, param=param).run()

paths = Func.PlotInitialize(frame_rate, "", sample_list, single_merge_compare=1, n_blocks=4).obtain_path()
Func.get_tau_info(sample_list, paths, frame_rate)
Func.get_clust_info(sample_list, paths)
# print(Func.read_pkl_info())
# print(Func.read_pkl_info(name="tau_info_SPAAC"))
