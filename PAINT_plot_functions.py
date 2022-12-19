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
import sympy as sym


class PlotInitialize:
    def __init__(self, frame_rate, save_name, samples, total_time_min=20, time_interval_min=5, all_params=True, param="",
                 x_list=None, x_label_name="", single_merge_compare=0, n_blocks=2):
        self.frame_rate = frame_rate
        self.x_list = x_list
        self.x_label_name = x_label_name
        self.save_name = save_name
        self.samples = samples

        initial_time = np.arange(1, total_time_min * 60 * frame_rate, time_interval_min * 60 * frame_rate)
        final_time = initial_time + time_interval_min * 60 * frame_rate - 1
        self.time_range = [str(initial_time[i]) + "-" + str(final_time[i]) for i in range(initial_time.size)]
        self.single_merge_compare = single_merge_compare
        self.n_blocks = n_blocks
        self.all_params = all_params
        self.param = param

    def obtain_path(self):
        if self.single_merge_compare == 0:
            paths = [self.samples[i] + self.time_range[0] for i in range(len(self.samples))]
        elif self.single_merge_compare == 1:
            if isinstance(self.samples, str):
                paths = self.samples + str(self.n_blocks) + "blocks"
            else:
                n_blocks = np.repeat(self.n_blocks, len(self.samples))
                paths = [self.samples[i] + str(n_blocks[i]) + "blocks" for i in range(len(self.samples))]
        else:
            n_blocks = np.arange(2, 5)
            paths = [self.samples + self.time_range[0]]
            for block in n_blocks:
                paths.append(self.samples + str(block) + "blocks")
            self.save_name = "_" + self.samples.split("_")[0] + self.save_name
        return paths

    def run(self, save=True):
        paths = PlotInitialize.obtain_path(self)

        param_list = ["tau_bright", "tau_bright_unfiltered", "tau_dark", "tau_dark_unfiltered", "density", "z_scores",
                      "density_filter_repeatedframe", "z_scores_filter_repeatedframe", "density_unfiltered",
                      "z_scores_unfiltered"]
        if self.all_params:
            for param in param_list:
                Plot(paths, self.frame_rate).run(param, True, x_list=self.x_list, x_label_name=self.x_label_name,
                                                 save_name=self.save_name)
        else:
            Plot(paths, self.frame_rate).run(self.param, save, x_list=self.x_list, x_label_name=self.x_label_name,
                                             save_name=self.save_name)


class Plot:
    def __init__(self, paths, framerate):
        self.paths_list = paths
        self.framerate = framerate

    def plot_hist_LT(self, data_arrays, save):
        fig, axs = plt.subplots(len(data_arrays), 1, sharex='all', sharey='all', figsize=(8, 6))  # , sharey='all'
        plt.subplots_adjust(hspace=0.4)
        i = 0
        for data in data_arrays:
            data = data.astype(int)
            # data_less_than_3s = data
            data_less_than_3s = data[data < 1.5 * self.framerate]
            print(data.size, data_less_than_3s.size, data_less_than_3s.size/data.size)
            axs[i].hist(data_less_than_3s / self.framerate)
            print(np.mean(data_less_than_3s / self.framerate))
            axs[i].set_title(self.paths_list[i], fontsize='small')
            i += 1
        fig.supylabel("Count")
        fig.supxlabel("Bound state lifetime (s)")
        if save:
            plt.savefig("bright_lifetime_hist.png", dpi=300)
            plt.clf()
            plt.close()
        else:
            plt.show()

    def plot_lifetime(self, data_arrays, exponential, dark_lt, save, x_list=None, save_name=""):
        colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
        # colors = cycle("rkkkkkk")
        if dark_lt:
            x_label = "Unbound state lifetime (s)"
            pre_save_name = "dark_time"
        else:
            x_label = "Bound state lifetime (s)"
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
            data = data.astype(int)[(data > 1)]
            tau, f, pred = obtain_pred_for_lifetimes(data, self.framerate, exponential)
            # plot
            plt.scatter(tau, 1 - f, lw=2, label=name, color=col, s=1)
            if exponential == 3:
                plt.plot(tau, exp_lifetimes3(tau, pred[0][0], pred[0][1], pred[0][2], pred[0][3], pred[0][4],
                                             pred[0][5]), "--", color=col)
                print(1 / pred[0][1], 1 / pred[0][3], 1 / pred[0][5], pred[0][0], 1 - pred[0][0] - pred[0][2],
                      1 - pred[0][0] - pred[0][2] - pred[0][4])
            elif exponential == 2:
                plt.plot(tau, exp_lifetimes2(tau, pred[0][0], pred[0][1], pred[0][2], pred[0][3]), "--", color=col)
                print(1 / pred[0][1], 1 / pred[0][3], pred[0][0], 1 - pred[0][0] - pred[0][2])
            else:
                plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]), "--", color=col)
                print(1 / pred[0][1], pred[0][0])

        # if not dark_lt:
        #     plt.xlim([0, 50])
        plt.ylim([0.01, 1])
        plt.xlabel(x_label)
        plt.ylabel('1 - cdf')
        plt.yscale('log')
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
        print(std_data / mean_data * 100)
        print(np.std(data_flat) / np.mean(data_flat) * 100)

        plt.plot(x, data_flat, ".")
        plt.errorbar(x_array, mean_data, yerr=std_data, fmt="x", capsize=5)

        if data_type == "z_scores" or data_type == "z_scores_filter_repeatedframe" or data_type == "z_scores_unfiltered":
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
        # plt.xscale("log")
        if x_list is None:
            str_ = [name.split("_")[0] for name in self.paths_list]
            plt.xticks(np.arange(0, np.max(x) + 1), str_)

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
                Plot.plot_lifetime(self, data, 3, False, True, x_list=x_list, save_name=save_name)
                # Plot.plot_hist_LT(self, data, False)
            elif data_type == "tau_bright_unfiltered":
                for path in self.paths_list:
                    data.append(get_all_data(path)[data_type])
                Plot.plot_lifetime(self, data, 2, False, save, x_list=x_list, save_name=save_name)
                # Plot.plot_hist_LT(self, data, save)
            elif data_type == "tau_dark":
                for path in self.paths_list:
                    data.append(get_all_data(path)[data_type])
                Plot.plot_lifetime(self, data, 2, True, save, x_list=x_list, save_name=save_name)
            elif data_type == "tau_dark_unfiltered":
                for path in self.paths_list:
                    data.append(get_all_data(path)[data_type])
                Plot.plot_lifetime(self, data, 2, True, save, x_list=x_list, save_name=save_name)
                # Plot.plot_hist_LT(self, data, save)
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
            data["tau_dark_unfiltered"] = np.load(path + "/IntensityCheck/all_unboundLT.npy")
            data["tau_bright_unfiltered"] = np.load(path + "/IntensityCheck/all_boundLT.npy")
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
        data["density_unfiltered"] = np.load(path + "/MoleculeDensity/density_multiple_roi_unfiltered.npy")
        data["density"] = np.load(path + "/MoleculeDensity/density_multiple_roi.npy")
        data["density_filter_repeatedframe"] = np.load(path + "/MoleculeDensity/density_multiple_roi_filter-repeated-frames.npy")
        data["z_scores_unfiltered"] = np.load(path + "/CETest/z_score_multiple_roi_unfiltered.npy")
        data["z_scores"] = np.load(path + "/CETest/z_score_multiple_roi.npy")
        data["z_scores_filter_repeatedframe"] = np.load(path + "/CETest/z_score_multiple_roi_filter-repeated-frames.npy")
    else:
        data["tau_bright"] = np.load(path + "/LifetimeAnalysis/all_bright_times.npy")
        data["tau_dark"] = np.load(path + "/LifetimeAnalysis/all_dark_times.npy")
        data["tau_dark_unfiltered"] = np.load(path + "/IntensityCheck/all_unboundLT.npy")
        data["tau_bright_unfiltered"] = np.load(path + "/IntensityCheck/all_boundLT.npy")
        data["n_clust_cent"] = np.load(path + "/MoleculeDensity/final_clust_center.npy").shape[0]
        data["n_clust_filtered"] = np.load(path + "/LifetimeAnalysis/clust_id_to_filter.npy").size
        data["n_clust_repeat"] = np.load(path + "/LifetimeAnalysis/trace_w_repeated_frames.npy").size
        data["density"] = np.load(path + "/MoleculeDensity/density.npy")
        data["z_scores"] = np.load(path + "/CETest/z_score.npy")
        data["density_filter_repeatedframe"] = np.load(path + "/MoleculeDensity/density_filter_filter-repeated-frames.npy")
        data["z_scores_filter_repeatedframe"] = np.load(path + "/CETest/z_score_filter-repeated-frames.npy")
        data["density_unfiltered"] = np.load(path + "/MoleculeDensity/density_unfiltered.npy")
        data["z_scores_unfiltered"] = np.load(path + "/CETest/z_score_unfiltered.npy")

    return data


def ecdf(sample_):
    # convert sample to a numpy array, if it isn't already
    sample_ = np.atleast_1d(sample_)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample_, return_counts=True)
    quantiles = np.insert(quantiles, 0, 0)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample_.size
    cumprob = np.insert(cumprob, 0, 0)
    # sample_sorted = np.sort(sample_)
    # cumprob = (np.arange(0, sample_sorted.size)) / sample_sorted.size

    return quantiles, cumprob


def exp_lifetimes(x, a, b):
    output = a * np.exp(-b * x)
    return output


def exp_lifetimes2(x, a, b, c, d):
    output = a * np.exp(-b * x) + ((1 - a - c) * np.exp(-d * x))
    return output


def exp_lifetimes2_noc(x, a, b, d):
    output = a * np.exp(-b * x) + ((1 - a) * np.exp(-d * x))
    return output


def exp_lifetimes3(x, a, b, c, d, e, f):
    output = a * np.exp(-b * x) + ((1 - a - c) * np.exp(-d * x)) + ((1 - a - c - e) * np.exp(-f * x))
    return output


def obtain_pred_for_lifetimes(lifetimes_nframes, framerate, exponential):
    tau, f = ecdf(lifetimes_nframes / framerate)
    y = 1 - f
    if exponential == 2:
        # Fit exponential fit - second constant = (1 - a - c)
        param_bounds = ([0, 0, 0, 0], [1, np.inf, 1, np.inf])
        pred = scipy.optimize.curve_fit(exp_lifetimes2, tau, y, np.array([0.1, 0.1, 0.1, 0.001]),
                                        bounds=param_bounds)

        # Fit exponential fit - second constant = (1 - a)
        # param_bounds = ([0, 0, 0], [1, np.inf, np.inf])
        # pred = scipy.optimize.curve_fit(exp_lifetimes2_noc, tau, y, np.array([0.1, 0.1, 0.001]), bounds=param_bounds)

    elif exponential == 3:
        param_bounds = ([0, 0, 0, 0, 0, 0], [1, np.inf, 1, np.inf, 1, np.inf])
        pred = scipy.optimize.curve_fit(exp_lifetimes3, tau, 1 - f, np.array([0.1, 0.1, 0.1, 0.001, 0.1, 0.001]),
                                        bounds=param_bounds)
    else:
        param_bounds = ([0, 0], [1, np.inf])
        pred = scipy.optimize.curve_fit(exp_lifetimes, tau, y, np.array([0.1, 0.1]), bounds=param_bounds)

    return tau, f, pred


def linear_func(x, a, m):
    output = a + m * x
    return output


def obtain_pred_from_lin_func(lifetimes_nframes, framerate, plot=False):
    tau, f = ecdf(lifetimes_nframes / framerate)
    y = 1 - f
    x = tau[y != 0]
    y = np.log(y[y != 0])

    def find_break_points(x_data, y_data, i=1, tol=0.01):
        x_diff, y_diff = np.diff(x_data), np.diff(y_data)
        grad_curr = np.abs(y_diff[0] / x_diff[0])
        break_pt = 0
        while i < x_diff.size:
            grad_next = np.abs(y_diff[i] / x_diff[i])
            if np.abs(grad_next - grad_curr) < tol:
                break_pt = i
                break
            else:
                grad_curr = grad_next
                i += 1
        if break_pt < 10:
            break_pt = 10
        return break_pt

    def find_outliers(y_data, tol=2.5):
        mean_y = np.mean(y_data)
        std_y = np.std(y_data)
        outliers = []
        i = 0
        for y_single in y_data:
            z_score = (y_single - mean_y) / std_y
            if np.abs(z_score) > tol:
                outliers.append(i)
            i += 1
        if not outliers:
            outliers = [y_data.size]
        return outliers

    break_ind = find_break_points(x, y, tol=0.3)
    break_ind2 = find_outliers(y[break_ind:])

    if plot:
        plt.scatter(x, y, lw=2, s=1)
        plt.scatter(x[break_ind:][break_ind2], y[break_ind:][break_ind2])

    param_bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
    pred1 = scipy.optimize.curve_fit(linear_func, x[:break_ind], y[:break_ind], np.array([0.1, 0.001]),
                                     bounds=param_bounds)
    frac1, tau1_ = np.exp(pred1[0][0]), -1 / pred1[0][1]
    if plot:
        plt.plot(x[:break_ind], linear_func(x[:break_ind], pred1[0][0], pred1[0][1]), "r--")

    pred2 = scipy.optimize.curve_fit(linear_func, x[break_ind:][:break_ind2[0]], y[break_ind:][:break_ind2[0]], np.array([0.1, 0.001]),
                                     bounds=param_bounds)
    frac2, tau2_ = np.exp(pred2[0][0]), -1 / pred2[0][1]

    if plot:
        plt.plot(x[break_ind:][:break_ind2[0]], linear_func(x[break_ind:][:break_ind2[0]], pred2[0][0], pred2[0][1]), "r--")
        # plt.plot(x, piecewise_linear(x, *popt), "r--")
        plt.show()
    return tau, f, [frac1, tau1_, frac2, tau2_]


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


def get_tau_info(samples, paths, frame_rate, exponential_b, exponential_ub):
    # Use PlotInitialize.obtain_path to get paths of the samples
    df = pd.DataFrame({'Samples': samples})
    tau_bright = np.zeros((len(samples), exponential_b * 2))
    tau_dark = np.zeros((len(samples), exponential_ub * 2))
    k = 0
    for path in paths:
        t = get_all_data(path, combine_roi=True)["tau_bright_unfiltered"]
        _, _, pred = obtain_pred_for_lifetimes(t[t > 1], frame_rate, exponential_b)
        tau_bright[k, :] = pred[0]

        t = get_all_data(path, combine_roi=True)["tau_dark"]
        _, _, pred = obtain_pred_for_lifetimes(t[t > 1], frame_rate, exponential_ub)
        tau_dark[k, :] = pred[0]
        k += 1
    df["frac_b_1"] = tau_bright[:, 0]
    df["tau_b_1"] = 1 / tau_bright[:, 1]
    df["frac_d"] = tau_dark[:, 0]
    df["tau_d"] = 1 / tau_dark[:, 1]

    if exponential_b == 2:
        df["frac_b_2"] = 1 - tau_bright[:, 0] - tau_bright[:, 2]
        df["tau_b_2"] = 1 / tau_bright[:, 3]
    elif exponential_b == 3:
        df["frac_b_3"] = 1 - tau_bright[:, 0] - tau_bright[:, 2] - tau_bright[:, 4]
        df["tau_b_3"] = 1 / tau_bright[:, 5]

    if exponential_b == 2:
        df["frac_d_2"] = 1 - tau_dark[:, 0] - tau_dark[:, 2]
        df["tau_d_2"] = 1 / tau_dark[:, 3]
    elif exponential_b == 3:
        df["frac_d_3"] = 1 - tau_dark[:, 0] - tau_dark[:, 2] - tau_dark[:, 4]
        df["tau_d_3"] = 1 / tau_dark[:, 5]

    df.to_pickle("tau_info.pkl")


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
    data_nonspec = read_pkl_info()
    samples = [sample.split("_")[0] for sample in data_nonspec["Samples"]]
    # x = np.arange(data_nonspec.shape[0])
    # labels = ["H24", "7H2", "H168"]
    x = np.arange(len(samples))
    tau = np.zeros((len(samples), 2))
    frac_arrays = np.zeros((len(samples), 2))
    for i in range(len(samples)):
        frac = np.append(data_nonspec["frac_b_1"][i], data_nonspec["frac_b_2"][i])
        tau_pre_sort = np.append(data_nonspec["tau_b_1"][i], data_nonspec["tau_b_2"][i])
        tau[i, :] = tau_pre_sort[np.argsort(frac)][::-1]
        frac_arrays[i, :] = frac[np.argsort(frac)][::-1]

    # for k in [1]:
    #     tau2, tau3 = tau[k, 1:]
    #     tau[k, 1:] = tau3, tau2
    #     frac2, frac3 = frac_arrays[k, 1:]
    #     frac_arrays[k, 1:] = frac3, frac2
    print(tau)

    plt.scatter(x, tau[:, 0], label="1")
    plt.scatter(x, tau[:, 1], label="2")
    # plt.scatter(x, tau[:, 2], label="3")
    plt.legend(loc="upper right")
    plt.xticks(x, samples)
    plt.ylim([0.1, 30])
    plt.yscale("log")
    plt.ylabel(r'$\tau_{bound}$' ' (s)')
    # plt.xlabel("SPAAC reaction duration (hr)")

    # fig, ax = plt.subplots(ncols=2, gridspec_kw={'wspace': 0.4})
    # data = [tau, frac_arrays]
    # for i in range(2):
    #     ax[i].scatter(x, data[i][:, 0], label="1")
    #     ax[i].scatter(x, data[i][:, 1], label="2")
    #     # ax[i].scatter(x, data[i][:, 2], label="3")
    #     ax[i].legend(loc="upper right")
    #     ax[i].set_xticks(x, samples)
    # ax[0].set_yscale("log")
    # ax[1].set_ylim(top=1)
    # ax[0].set_ylabel(r'$\tau_{bound}$' ' (s)')
    # ax[1].set_ylabel("Fraction of bound lifetimes")
    # # ax[0].xlabel("SPAAC reaction duration (hr)")
    if save:
        plt.savefig("tau_frac_bound.png", dpi=300)
    else:
        plt.show()


def get_density_z_info(samples, paths, data_type=""):
    # Use PlotInitialize.obtain_path to get paths of the samples
    df = pd.DataFrame({'Samples': samples})
    density = np.zeros((len(samples), 5))
    z_scores = np.zeros((len(samples), 5))
    if data_type == "unfiltered":
        saved_name = "_unfiltered"
    elif data_type == "repeat":
        saved_name = "_filter_repeatedframe"
    else:
        saved_name = data_type

    k = 0
    for path in paths:
        data = get_all_data(path, combine_roi=True)
        density[k, :] = data["density" + saved_name]
        z_scores[k, :] = data["z_scores" + saved_name]
        k += 1

    for i in range(5):
        name = "density_" + str(i)
        df[name] = density[:, i]

    for i in range(5):
        name = "z-scores_" + str(i)
        df[name] = z_scores[:, i]

    df.to_pickle("density_z_info.pkl")


def pickle_to_data_array(data, data_type, choose_data=None):
    if choose_data is None:
        choose_data = range(data.shape[0])
    name = data_type + "_"
    data_1_array = np.array([])
    for i in range(5):
        name_i = name + str(i)
        data_1_array = np.append(data_1_array, data[name_i][choose_data[0]])
    data_array = [data_1_array]

    for data_index in choose_data[1:]:
        data_1_array = np.array([])
        for i in range(5):
            name_i = name + str(i)
            data_1_array = np.append(data_1_array, data[name_i][data_index])
        data_array.append(data_1_array)
    return data_array


def plot_density_z(save=True, save_name=""):

    def neg_exp_lifetimes(x_data, a, b, d):
        output = - a * np.exp(-b * x_data) + d
        return output

    def flatten_data(data_array):
        data_flattened = np.array([])
        for i in range(len(data_array)):
            data_flattened = np.append(data_flattened, data_array[i])
        return data_flattened

    def mean_squared_error(x_exp, exp_y, fitted_y, min_x_fit, max_x_fit):
        x_for_fit = np.arange(min_x_fit, max_x_fit, 0.1)
        squared_error = np.zeros(x_exp.size)
        k = 0
        for x_single in x_exp:
            index = (np.abs(x_single - x_for_fit) < 1e-10)
            y_single = exp_y[k]
            squared_error[k] = (y_single - fitted_y[index][0]) ** 2
            k += 1
        return np.mean(squared_error)

    data = read_pkl_info(name="density_z_info")
    data_type = "z-scores"  # "z-scores", "density"
    data_mult_arrays = [pickle_to_data_array(data, data_type, choose_data=np.arange(0, 4)),
                        pickle_to_data_array(data, data_type, choose_data=np.arange(4, 8)),
                        pickle_to_data_array(data, data_type, choose_data=np.arange(8, 12))]
    x_arrays = [[1, 1 * 24, 3 * 24, 7 * 24], [1, 1 * 24, 3 * 24, 7 * 24], [1, 1 * 24, 3 * 24, 7 * 24]]
    # data_mult_arrays = [pickle_to_data_array(data, data_type, choose_data=np.arange(0, 4)),
    #                     pickle_to_data_array(data, data_type, choose_data=np.arange(4, 8)),
    #                     pickle_to_data_array(data, data_type, choose_data=np.arange(8, 12))]
    # x_arrays = [[1, 1 * 24, 3 * 24, 7 * 24], [1, 1 * 24, 3 * 24, 7 * 24], [1, 1 * 24, 3 * 24, 7 * 24]]
    xlabel_name = "SPAAC reaction duration (hr)"
    data_type = data_type.replace("-", "_")

    if "density" in data_type:
        ylabel_name = "Density (\u03BC$m^{-2}$)"
        x_plot_min = 0
        if data_type == "density_filter_repeatedframe":
            title_name = "Clusters with repeated frames are filtered!"
        else:
            title_name = ""
    else:
        ylabel_name = "z-scores"
        x_plot_min = np.min(x_arrays[0])
        if data_type == "z_scores_filter_repeatedframe":
            title_name = "Clusters with repeated frames are filtered!!"
        else:
            title_name = ""

    col = ["tab:blue", "tab:orange", "tab:red"]
    # labels = ["10%N3", "1%N3", "10%N3-mismatch"]
    labels = [str(i) for i in range(3)]

    val = []
    mean = np.array([])
    for c in range(3):
        data_flat = flatten_data(data_mult_arrays[c])
        x, mean_data, std_data = obtain_mean_std_of_data(data_mult_arrays[c], x_arrays[c])
        mean = np.append(mean, mean_data)
        param_bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
        pred = scipy.optimize.curve_fit(neg_exp_lifetimes, x, data_flat, np.array([200, 0.1, 200]), bounds=param_bounds,
                                        max_nfev=500 * x.size)
        print(pred[0][0], 1 / pred[0][1], pred[0][2])
        val.append(1 / pred[0][1])
        # plt.plot(x, data_flat, ".", label=labels[c], color=col[c])
        y = neg_exp_lifetimes(np.arange(x_plot_min, np.max(x) + 1, 0.1), pred[0][0], pred[0][1], pred[0][2])
        print(mean_squared_error(x, data_flat, y, x_plot_min, np.max(x) + 1))
        plt.plot(np.arange(x_plot_min, np.max(x) + 1, 0.1), y, "--", color=col[c])
        plt.errorbar(x_arrays[c], mean_data, yerr=std_data, fmt="x", capsize=5, c=col[c], label=labels[c])
    print(np.mean(val))
    # for c in [2]:
    #     data_arrays = data_mult_arrays[c]
    #     x_array = x_arrays[c]
    #
    #     data_flat = flatten_data(data_arrays)
    #     x, mean_data, std_data = obtain_mean_std_of_data(data_arrays, x_array)
    #
    #     plt.plot(x, data_flat, ".", label=labels[c], color=col[c])
    #     param_bounds = ([0, 0], [500, np.inf])
    #     pred = scipy.optimize.curve_fit(exp_lifetimes, x, data_flat, np.array([200, 0.1]), bounds=param_bounds)
    #     print(pred[0][0], 1 / pred[0][1])
    #     y = exp_lifetimes(np.arange(x_plot_min, np.max(x) + 1, 0.1), pred[0][0], pred[0][1])
    #     print(mean_squared_error(x, data_flat, y, x_plot_min, np.max(x) + 1))
    #     plt.plot(np.arange(x_plot_min, np.max(x) + 1, 0.1), y, "--", color=col[c])
    #     plt.errorbar(x_array, mean_data, yerr=std_data, fmt="x", capsize=5, c="k")

    # k = 1
    # labels = ["1%N3", "10%N3", "-ve control"]
    # for data_arrays in data_mult_arrays[1:]:
    #     data_flat = data_arrays[0]
    #     x = np.repeat(x_arrays[k], data_flat.size)
    #     mean_data = np.append(mean_data, np.mean(data_flat))
    #     std_data = np.append(std_data, np.std(data_flat))
    #
    #     plt.plot(x, data_flat, ".", label=labels[k-1])
    #     plt.errorbar(x_arrays[k], np.mean(data_flat), yerr=np.std(data_flat), fmt="x", capsize=5, c="k")
    #     k += 1

    if data_type == "z_scores" or data_type == "z_scores_filter_repeatedframe":
        plt.axhline(y=1.65, ls="--", c="k")
        plt.axhline(y=-1.65, ls="--", c="k")
        # if np.count_nonzero((mean_data < -4) | (mean_data > 4)) == 0:
        #     plt.ylim([-4, 4])
    else:
        if np.max(mean) > 5:
            plt.ylim(bottom=-3)
            # pass
        if "filter" in data_type:
            plt.title(title_name)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.legend()

    if save:
        plt.savefig(data_type + save_name + ".png", dpi=300)
        plt.clf()
    else:
        plt.show()


def plot_density_bar_plot(save=True):
    data = read_pkl_info(name="density_z_info")
    save_name = ""
    samples = [sample.split("_")[0] for sample in data["Samples"]]
    x_pos = np.arange(len(samples))
    data_arrays = pickle_to_data_array(data, "density")
    mean_data, std_data = obtain_mean_std_of_data(data_arrays, x_pos)[1:]

    plt.bar(x_pos, mean_data, yerr=std_data, align="center", ecolor='black', capsize=5, width=0.35)
    plt.ylabel("Density (\u03BC$m^{-2}$)")
    plt.xticks(x_pos, samples)
    # plt.yaxis.grid(True)

    if save:
        plt.savefig("density_bar" + save_name + ".png", dpi=300)
        plt.clf()
    else:
        plt.show()


def plot_locs_per_cluster(paths, save=True):
    # sample_list = ["1%N3H72_", "1%N3noncomp_"]
    data_multi = []
    for path in paths:
        labels = np.array([])
        for i in range(5):
            ms_labels = np.load(path + "/Clustering/roi" + str(i) + "/ms_labels.npy")
            labels = np.append(labels, ms_labels)
        _, count = np.unique(labels, return_counts=True)
        data_multi.append(count[count < 200])
    plt.hist(data_multi, density=True, histtype='bar', stacked=True, bins=25,
             label=["9bp complementarity", "9bp mismatch"])
    plt.xlabel("Number of localizations per cluster")
    plt.ylabel("Normalized frequency")
    plt.legend()
    plt.xlim([-1, 200])
    plt.title("1%N3")
    if save:
        plt.savefig("locs_per_cluster.png", dpi=300)
    else:
        plt.show()
