import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import logging
from itertools import cycle
from time import time
import os
from tqdm import tqdm
import scipy


class PlotInitialize:
    def __init__(self, frame_rate, save_name, samples, all_params=True, param="", x_list=None, x_label_name="",
                 single_merge_compare=0, n_blocks=2):
        self.frame_rate = frame_rate
        self.x_list = x_list
        self.x_label_name = x_label_name
        self.save_name = save_name
        self.samples = samples

        initial_time = np.arange(1, 20 * 60 * frame_rate, 5 * 60 * frame_rate)
        final_time = initial_time + 5 * 60 * frame_rate - 1
        self.time_range = [str(initial_time[i]) + "-" + str(final_time[i]) for i in range(initial_time.size)]
        self.single_merge_compare = single_merge_compare
        self.n_blocks = n_blocks
        self.all_params = all_params
        self.param = param

    def obtain_path(self):
        if self.single_merge_compare == 0:
            paths = [self.samples[i] + self.time_range[0] for i in range(len(self.samples))]
        elif self.single_merge_compare == 1:
            n_blocks = np.repeat(self.n_blocks, len(self.samples))
            paths = [self.samples[i] + str(n_blocks[i]) + "blocks" for i in range(len(self.samples))]
        else:
            n_blocks = np.arange(2, 5)
            paths = [self.samples + self.time_range[0]]
            for block in n_blocks:
                paths.append(self.samples + str(block) + "blocks")
            self.save_name = "_" + self.samples.split("_")[0] + self.save_name
        return paths

    def run(self):
        paths = PlotInitialize.obtain_path(self)

        param_list = ["tau_bright", "tau_dark", "density", "z_scores", "density_filter_repeatedframe",
                      "z_scores_filter_repeatedframe"]
        if self.all_params:
            for param in param_list:
                Plot(paths, self.frame_rate).run(param, True, x_list=self.x_list, x_label_name=self.x_label_name,
                                                 save_name=self.save_name)
        else:
            Plot(paths, self.frame_rate).run(self.param, True, x_list=self.x_list, x_label_name=self.x_label_name,
                                             save_name=self.save_name)


class Plot:
    def __init__(self, paths, framerate):
        self.paths_list = paths
        self.framerate = framerate

    def plot_lifetime(self, data_arrays, exponential, dark_lt, save, x_list=None, save_name=""):
        colors = cycle("bgrcmybgrcmybgrcmybgrcmy")
        if dark_lt:
            x_label = r'$\tau_{off}$' ' (s)'
            pre_save_name = "dark_time"
        else:
            x_label = r'$\tau_{on}$' ' (s)'
            pre_save_name = "bright_time"

        if len(self.paths_list) != len(data_arrays):
            legend_list = ["roi" + str(i) for i in range(5)]
            title_name = self.paths_list
            pre_save_name = self.paths_list + "_" + pre_save_name
        else:
            if x_list is None:
                legend_list = [name.split("_")[0] for name in self.paths_list]
            else:
                if isinstance(x_list, str):
                    legend_list = x_list
                else:
                    legend_list = [str(x) for x in x_list]
            title_name = ""

        for data, col, name in zip(data_arrays, colors, legend_list):
            tau, f, pred = obtain_pred_for_lifetimes(data, self.framerate, exponential)
            # plot
            plt.scatter(tau, 1 - f, lw=2, label=name, color=col, s=1)
            if exponential == 2:
                plt.plot(tau, exp_lifetimes2(tau, pred[0][0], pred[0][1], pred[0][2], pred[0][3]), "--", color=col)
            else:
                plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]), "--", color=col)

        if dark_lt:
            plt.ylim(bottom=0.01)
        else:
            plt.ylim(bottom=0.0001)
        plt.xlabel(x_label)
        plt.ylabel('1 - cdf')
        plt.yscale('log', nonpositive="clip")
        plt.legend(fancybox=True, loc='upper right')
        plt.title(title_name)

        if save:
            plt.savefig(pre_save_name + save_name + ".png", dpi=300)
            plt.clf()
            plt.close()
        else:
            plt.show()

    def plot_clust_info(self, data_arrays, save, x_list=None, x_label_name="", save_name=""):
        n_clust_cent = np.array([])
        n_clust_filtered = np.array([])

        if x_list is None:
            x_array = range(len(data_arrays) // 2)
            xlabel_name = "Sample"
        else:
            x_array = x_list
            xlabel_name = x_label_name

        for i in np.arange(0, len(data_arrays), 2):
            n_clust_cent = np.append(n_clust_cent, data_arrays[i] + data_arrays[i+1])
        for i in np.arange(1, len(data_arrays), 2):
            n_clust_filtered = np.append(n_clust_filtered, data_arrays[i])

        x = np.array([])
        k = 0
        for i in x_array:
            if isinstance(data_arrays[k], int):
                x = np.append(x, np.repeat(i, 1))
            else:
                x = np.append(x, np.repeat(i, len(data_arrays[k])))
            k += 2

        plt.plot(x, n_clust_cent, ".", label="Total clusters")
        plt.plot(x, n_clust_filtered, ".", label="Filtered clusters")

        plt.xlabel(xlabel_name)
        plt.ylabel("Count")
        if x_list is None:
            plt.xticks(np.arange(0, np.max(x) + 1), self.paths_list)
        plt.legend()
        if save:
            plt.savefig("clust_info" + save_name + ".png", dpi=300)
            plt.clf()
        else:
            plt.show()

        plt.plot(x, n_clust_filtered/n_clust_cent, ".")

        plt.ylim([0, 1])
        plt.xlabel(xlabel_name)
        plt.ylabel("Filtered clusters/Total clusters")
        if x_list is None:
            plt.xticks(np.arange(0, np.max(x) + 1), self.paths_list)

        if save:
            plt.savefig("clust_info_ratio.png", dpi=300)
            plt.clf()
        else:
            plt.show()

    def plot_density_or_z_scores(self, data_type, data_arrays, save, x_list=None, x_label_name="", save_name=""):
        if x_list is None:
            x_array = range(len(data_arrays))
            xlabel_name = "Sample"
        else:
            x_array = x_list
            xlabel_name = x_label_name

        if "density" in data_type:
            ylabel_name = "Density (\u03BC$m^{-2}$)"
            if data_type == "density_filter_repeatedframe":
                title_name = "Clusters with repeated frames are filtered!"
            else:
                title_name = ""
        else:
            ylabel_name = "z-scores"
            if data_type == "z_scores_filter_repeatedframe":
                title_name = "Clusters with repeated frames are filtered!!"
            else:
                title_name = ""

        data_flat = np.array([])
        for i in range(len(data_arrays)):
            data_flat = np.append(data_flat, data_arrays[i])

        x, mean_data, std_data = obtain_mean_std_of_data(data_arrays, x_array)

        plt.plot(x, data_flat, ".")
        plt.errorbar(x_array, mean_data, yerr=std_data, fmt="x", capsize=5)

        if data_type == "z_scores" or data_type == "z_scores_filter_repeatedframe":
            plt.axhline(y=1.65, ls="--", c="k")
            plt.axhline(y=-1.65, ls="--", c="k")
            if np.count_nonzero((mean_data < -4) | (mean_data > 4)) == 0:
                plt.ylim([-4, 4])
        else:
            if np.max(mean_data) > 5:
                plt.ylim(bottom=-3)
            if "filter" in data_type:
                plt.title(title_name)
        plt.xlabel(xlabel_name)
        plt.ylabel(ylabel_name)
        if x_list is None:
            str = [name.split("_")[0] for name in self.paths_list]
            plt.xticks(np.arange(0, np.max(x) + 1), str)

        if save:
            plt.savefig(data_type + save_name + ".png", dpi=300)
            plt.clf()
        else:
            plt.show()

    def run(self, data_type, save, x_list=None, x_label_name="", save_name=""):
        data = []
        if isinstance(self.paths_list, str):
            if data_type == "tau_bright":
                get_data = get_all_data(self.paths_list, combine_roi=False)
                for i in range(5):
                    data.append(get_data["tau_bright_roi" + str(i)])
                Plot.plot_lifetime(self, data, 2, False, save, save_name=save_name)
            elif data_type == "tau_dark":
                get_data = get_all_data(self.paths_list, combine_roi=False)
                for i in range(5):
                    data.append(get_data["tau_dark_roi" + str(i)])
                Plot.plot_lifetime(self, data, 1, True, save, save_name=save_name)
        else:
            if data_type == "tau_bright":
                for path in self.paths_list:
                    data.append(get_all_data(path)[data_type])
                Plot.plot_lifetime(self, data, 2, False, save, x_list=x_list, save_name=save_name)
            elif data_type == "tau_dark":
                for path in self.paths_list:
                    data.append(get_all_data(path)[data_type])
                Plot.plot_lifetime(self, data, 1, True, save, x_list=x_list, save_name=save_name)
            elif data_type == "clust_info":
                for path in self.paths_list:
                    data.append(get_all_data(path)["n_clust_cent"])
                    data.append(get_all_data(path)["n_clust_filtered"])
                Plot.plot_clust_info(self, data, save, x_list=x_list, x_label_name=x_label_name, save_name=save_name)
            else:
                for path in self.paths_list:
                    data.append(get_all_data(path)[data_type])
                Plot.plot_density_or_z_scores(self, data_type, data, save, x_list=x_list, x_label_name=x_label_name,
                                              save_name=save_name)


def get_tau_info(samples, paths, frame_rate):
    # Use PlotInitialize.obtain_path to get paths of the samples
    df = pd.DataFrame({'Samples': samples})
    tau_bright = np.zeros((len(samples), 4))
    tau_dark = np.zeros((len(samples), 2))
    k = 0
    for path in paths:
        t = get_all_data(path, combine_roi=True)["tau_bright"]
        _, _, pred = obtain_pred_for_lifetimes(t, frame_rate, 2)
        tau_bright[k, :] = pred[0][0], 1 / pred[0][1], 1 - pred[0][0] - pred[0][2], 1 / pred[0][3]

        t = get_all_data(path, combine_roi=True)["tau_dark"]
        _, _, pred = obtain_pred_for_lifetimes(t, frame_rate, 1)
        tau_dark[k, :] = pred[0][0], 1 / pred[0][1]
        k += 1
    df["frac_b_1"] = tau_bright[:, 0]
    df["tau_b_1"] = tau_bright[:, 1]
    df["frac_b_2"] = tau_bright[:, 2]
    df["tau_b_2"] = tau_bright[:, 3]

    df["frac_d"] = tau_dark[:, 0]
    df["tau_d"] = tau_dark[:, 1]
    df.to_pickle("tau_info.pkl")


def get_all_data(path, combine_roi=True):
    data = {}
    if os.path.exists(path + "/Clustering/roi0"):
        if combine_roi:
            tau_bright = np.array([])
            tau_dark = np.array([])
            for i in range(5):
                tau_bright = np.append(tau_bright, np.load(path + "/LifetimeAnalysis/roi" + str(i) + "/all_bright_times.npy"))
                tau_dark = np.append(tau_dark, np.load(path + "/LifetimeAnalysis/roi" + str(i) + "/all_dark_times.npy"))
            data["tau_bright"] = tau_bright
            data["tau_dark"] = tau_dark
        else:
            for i in range(5):
                data["tau_bright_roi" + str(i)] = np.load(path + "/LifetimeAnalysis/roi" + str(i) + "/all_bright_times.npy")
                data["tau_dark_roi" + str(i)] = np.load(path + "/LifetimeAnalysis/roi" + str(i) + "/all_dark_times.npy")
        n_clust_filtered = np.array([])
        n_clust_cent = np.array([])
        n_clust_repeat = np.array([])
        for i in range(5):
            clust_id = np.load(path + "/LifetimeAnalysis/roi" + str(i) + "/clust_id_to_filter.npy")
            clust_cent = np.load(path + "/MoleculeDensity/final_clust_center" + str(i) + ".npy")
            clust_repeat = np.load(path + "/LifetimeAnalysis/roi" + str(i) + "/trace_w_repeated_frames.npy")
            n_clust_filtered = np.append(n_clust_filtered, clust_id.size)
            n_clust_cent = np.append(n_clust_cent, clust_cent.shape[0])
            n_clust_repeat = np.append(n_clust_repeat, clust_repeat.size)
        data["n_clust_repeat"] = n_clust_repeat
        data["n_clust_cent"] = n_clust_cent
        data["n_clust_filtered"] = n_clust_filtered
        data["density"] = np.load(path + "/MoleculeDensity/density_multiple_roi.npy")
        data["density_filter_repeatedframe"] = np.load(path + "/MoleculeDensity/density_multiple_roi_filter-repeated-frames.npy")
        data["z_scores"] = np.load(path + "/CETest/z_score_multiple_roi.npy")
        data["z_scores_filter_repeatedframe"] = np.load(path + "/CETest/z_score_multiple_roi_filter-repeated-frames.npy")
    else:
        data["tau_bright"] = np.load(path + "/LifetimeAnalysis/all_bright_times.npy")
        data["tau_dark"] = np.load(path + "/LifetimeAnalysis/all_dark_times.npy")
        data["n_clust_cent"] = np.load(path + "/MoleculeDensity/final_clust_center.npy").shape[0]
        data["n_clust_filtered"] = np.load(path + "/LifetimeAnalysis/clust_id_to_filter.npy").size
        data["n_clust_repeat"] = np.load(path + "/LifetimeAnalysis/trace_w_repeated_frames.npy").size
        data["density"] = np.load(path + "/MoleculeDensity/density.npy")
        data["z_scores"] = np.load(path + "/CETest/z_score.npy")
        data["density_filter_repeatedframe"] = np.load(path + "/MoleculeDensity/density_filter-repeated-frames.npy")
        data["z_scores_filter_repeatedframe"] = np.load(path + "/CETest/z_score_filter-repeated-frames.npy")

    return data


def ecdf(sample_):
    # convert sample to a numpy array, if it isn't already
    sample_ = np.atleast_1d(sample_)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample_, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample_.size

    return quantiles, cumprob


def exp_lifetimes(x, a, b):
    output = a * np.exp(-b * x)
    return output


def exp_lifetimes2(x, a, b, c, d):
    output = a * np.exp(-b * x) + ((1 - a - c) * np.exp(-d * x))
    return output


def obtain_pred_for_lifetimes(lifetimes_nframes, framerate, exponential):
    tau, f = ecdf(lifetimes_nframes / framerate)
    y = 1 - f
    if exponential == 2:
        param_bounds = ([0, 0, 0, 0], [1, np.inf, 1, np.inf])
        pred = scipy.optimize.curve_fit(exp_lifetimes2, tau, y, np.array([0.1, 0.1, 0.1, 0.001]),
                                        bounds=param_bounds)
    else:
        param_bounds = ([0, 0], [1, np.inf])
        pred = scipy.optimize.curve_fit(exp_lifetimes, tau, y, np.array([0.1, 0.1]), bounds=param_bounds)

    return tau, f, pred


def obtain_mean_std_of_data(data_arrays, x_array):
    x = np.array([])
    mean_data = np.array([])
    std_data = np.array([])
    k = 0
    for i in x_array:
        x = np.append(x, np.repeat(i, data_arrays[k].size))
        if data_arrays[k].size != 1:
            mean_data = np.append(mean_data, np.mean(data_arrays[k]))
            std_data = np.append(std_data, np.std(data_arrays[k]))
        else:
            mean_data = np.append(mean_data, data_arrays[k])
            std_data = np.append(std_data, 0)
        k += 1

    return x, mean_data, std_data


def read_pkl_info(name="tau_info"):
    data = pd.read_pickle(name + ".pkl")
    return data


def get_clust_info(samples, paths):
    # Use PlotInitialize.obtain_path to get paths of the samples
    df = pd.DataFrame({'Samples': samples})
    ratio_clust_filtered = np.zeros((len(samples), 2))
    ratio_clust_repeat = np.zeros((len(samples), 2))
    k = 0
    for path in paths:
        data = get_all_data(path, combine_roi=True)
        filtered = data["n_clust_filtered"] / (data["n_clust_cent"] + data["n_clust_filtered"])
        repeat = data["n_clust_repeat"] / (data["n_clust_cent"] + data["n_clust_filtered"])
        ratio_clust_filtered[k, :] = np.min(filtered), np.max(filtered)
        ratio_clust_repeat[k, :] = np.min(repeat), np.max(repeat)
        k += 1

    df["filtered_min"] = ratio_clust_filtered[:, 0]
    df["filtered_max"] = ratio_clust_filtered[:, 1]
    df["repeat_min"] = ratio_clust_repeat[:, 0]
    df["repeat_max"] = ratio_clust_repeat[:, 1]

    df.to_pickle("clust_info.pkl")


def plot_tau(save=True):
    data_nonspec = Func.read_pkl_info()
    data_SPAAC = Func.read_pkl_info(name="tau_info_SPAAC")
    tau_type = "tau_b_2"
    plt.scatter([1, 24, 72, 168], data_SPAAC[tau_type], label="10%N3")
    plt.scatter([72], data_nonspec[tau_type][1], label="1%N3")
    plt.scatter([-10, -10], [data_nonspec[tau_type][0], data_nonspec[tau_type][3]], label="-ve control")
    plt.legend()
    plt.ylabel(r'$\tau_{on, 2}$' ' (s)')
    plt.xlabel("SPAAC reaction duration (hr)")
    if save:
        plt.savefig("tau_on_2.png", dpi=300)
    else:
        plt.show()
