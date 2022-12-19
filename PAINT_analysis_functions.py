import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from sklearn.cluster import MeanShift, Birch, MiniBatchKMeans, DBSCAN
from itertools import cycle
from time import time
import os
from tqdm import tqdm
import scipy
import sys


class PAINT:

    def __init__(self, name, path):
        self.name = name
        self.val = pd.read_csv(self.name + ".csv").values
        self.id = self.val[:, 0]
        self.frame = self.val[:, 1] - 1
        self.x = self.val[:, 2]  # nm
        self.y = self.val[:, 3]  # nm
        self.sigma = self.val[:, 4]  # nm
        self.intensity = self.val[:, 5]  # photons
        self.offset = self.val[:, 6]  # photon
        self.bkgstd = self.val[:, 7]  # photon
        self.uncertainty = self.val[:, 8]  # nm
        self.n_frame = np.unique(self.frame).size
        self.path = path + "/Raw"

    def plot_coord_raw(self, zoom=5000):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        plt.scatter(self.x, self.y, s=1)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.title(self.name + ", s=1")
        plt.savefig(self.path + "/coords_raw.png", dpi=300)
        plt.clf()
        plt.close()

        fig, _ = plt.subplots(figsize=(6, 6))
        plt.scatter(self.x, self.y, s=0.5)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.title(self.name + ", filtered, s=1")
        plt.xlim([50000 / 2 - zoom, 50000 / 2 + zoom])
        plt.ylim([60000 / 2 - zoom, 60000 / 2 + zoom])
        plt.title(self.name + "zoomed, s=1")
        plt.savefig(self.path + "/coords_raw_zoomed.png", dpi=300)
        plt.clf()
        plt.close()

    def plot_prop_raw(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        plt.hist(self.sigma, bins=100, density=1)
        plt.xlabel("Sigma (nm)")
        plt.title("Sigma\n" + self.name + ", bins=100")
        plt.savefig(self.path + "/sigma.png", dpi=300)
        plt.clf()

        plt.hist(self.intensity, bins=100, density=1)
        plt.xlabel("Intensity (photon)")
        plt.title("Intensity\n" + self.name + ", bins=100")
        plt.savefig(self.path + "/intensity.png", dpi=300)
        plt.clf()

        plt.hist(self.offset, bins=100, density=1)
        plt.xlabel("Offset (photon)")
        plt.title("Offset\n" + self.name + ", bins=100")
        plt.savefig(self.path + "/offset.png", dpi=300)
        plt.clf()

        plt.hist(self.uncertainty, bins=100, density=1)
        plt.xlabel("Uncertainty (nm)")
        plt.title("Uncertainty\n" + self.name + ", bins=100")
        plt.savefig(self.path + "/uncertainty.png", dpi=300)
        plt.clf()
        plt.close()


class PAINTFiltered(PAINT):

    def __init__(self, name, path):
        PAINT.__init__(self, name, path)
        self.path = path + "/PreProcessing"
        self.filtered_id = np.array([])
        self.loc_precision = 0
        self.path_drift = path + "/DriftCorrection"
        self.ratio_filter = np.array([])

    def filter_uncertainty(self, plt_save=False, update_prop=False):
        # if max(self.uncertainty) > 50:
        #     mu, sigma = norm.fit(self.uncertainty)
        #     filtered_id = np.where(self.uncertainty < mu + s * sigma)[0]
        #     f_uncertainty = self.uncertainty[filtered_id]
        #     print("Uncertainty filtering: s=4")
        #     print("Uncertainty > " + str(mu + s * sigma) + " nm filtered")
        #     print("Total " + str(self.uncertainty.size - filtered_id.size) + " data points filtered")
        #     mu, sigma = norm.fit(f_uncertainty)
        #     print("Localization precision = " + str(mu) + " nm")
        #
        #     plt.hist(f_uncertainty, bins=100, density=1)
        #     x = np.linspace(f_uncertainty.min(), f_uncertainty.max(), 1000)
        #     plt.plot(x, norm.pdf(x, mu, sigma))
        med = np.median(self.uncertainty)
        upper_bound = 2 * med
        filtered_id = np.where(self.uncertainty < upper_bound)[0]
        f_uncertainty = self.uncertainty[filtered_id]
        print("Uncertainty > " + str(upper_bound) + " nm filtered")

        plt.hist(f_uncertainty, bins=100, density=1)
        plt.axvline(x=upper_bound, ls="--")
        plt.xlabel("Uncertainty/Precision (nm)")
        plt.title("Uncertainty filtering\n" + self.name + ", bins=100")
        if plt_save:
            plt.savefig(self.path + "/filter_uncertainty.png", dpi=300)
            plt.clf()
            plt.close()
        else:
            # plt.show()
            plt.clf()
            plt.close()

        if update_prop:
            self.filtered_id = np.append(self.filtered_id, filtered_id)

    def filter_intensity(self, s=4, plt_save=False, update_prop=False):
        mu, sigma = norm.fit(self.intensity)
        filtered_id = np.where(self.intensity < mu + s * sigma)[0]
        f_intensity = self.intensity[filtered_id]
        print("Intensity filtering: s=4")
        print("Intensity > " + str(mu + s * sigma) + " photons filtered")
        print("Total " + str(self.intensity.size - filtered_id.size) + " data points filtered")
        mu, sigma = norm.fit(f_intensity)

        plt.hist(f_intensity, bins=100, density=1)
        x = np.linspace(f_intensity.min(), f_intensity.max(), 1000)
        plt.plot(x, norm.pdf(x, mu, sigma))
        plt.xlabel("Intensity (photon)")
        plt.title("Intensity filtering\n" + self.name + ", bins=100")
        if plt_save:
            plt.savefig(self.path + "/filter_intensity.png", dpi=300)
            plt.clf()
            plt.close()
        else:
            # plt.show()
            plt.clf()
            plt.close()

        if update_prop:
            self.filtered_id = np.append(self.filtered_id, filtered_id)

    def filter_sigma(self, plt_save=False, update_prop=False):
        # mu, sigma = norm.fit(self.sigma)
        # filtered_id = np.where((self.sigma < mu + s * sigma) & (self.sigma > mu - s * sigma))[0]
        # f_sigma = self.sigma[filtered_id]
        # logging.info("Sigma filtering: s=4")
        # logging.info("Sigma > " + str(mu + s * sigma) + " nm and sigma < " + str(mu - s * sigma) + " nm filtered")
        # logging.info("Total " + str(self.sigma.size - filtered_id.size) + " data points filtered")
        # mu, sigma = norm.fit(f_sigma)
        #
        # plt.hist(f_sigma, bins=100, density=1)
        # x = np.linspace(f_sigma.min(), f_sigma.max(), 1000)
        # plt.plot(x, norm.pdf(x, mu, sigma))

        med = np.median(self.sigma)
        upper_bound, lower_bound = med * 1.5, med * 0.5
        filtered_id = np.where((self.sigma < upper_bound) & (self.sigma > lower_bound))[0]
        f_sigma = self.sigma[filtered_id]
        print("Sigma > " + str(upper_bound) + " nm and sigma < " + str(lower_bound) + " nm filtered")

        plt.hist(f_sigma, bins=100, density=1)
        plt.axvline(x=upper_bound, ls="--")
        plt.axvline(x=lower_bound, ls="--")
        plt.xlabel("Sigma (nm)")
        plt.title("Sigma filtering\n" + self.name + ", bins=100")
        if plt_save:
            plt.savefig(self.path + "/filter_sigma.png", dpi=300)
            plt.clf()
            plt.close()
        else:
            # plt.show()
            plt.clf()
            plt.close()

        if update_prop:
            self.filtered_id = np.append(self.filtered_id, filtered_id)

    def filter_offset(self, plt_save=False, update_prop=False):
        # mu, sigma = norm.fit(self.offset)
        # filtered_id = np.where((self.offset < mu + s * sigma) & (self.offset > mu - s * sigma) & (self.offset > 0))[0]
        # f_offset = self.offset[filtered_id]
        # logging.info("Offset filtering: s=2 and offset > 0")
        # logging.info("Offset > " + str(mu + s * sigma) + " photons and offset < " + str(mu - s * sigma) + " filtered")
        # logging.info("Total " + str(self.offset.size - filtered_id.size) + " data points filtered")
        # mu, sigma = norm.fit(f_offset)
        #
        # plt.hist(f_offset, bins=100, density=1)
        # x = np.linspace(f_offset.min(), f_offset.max(), 1000)
        # plt.plot(x, norm.pdf(x, mu, sigma))
        med = np.median(self.offset)
        upper_bound, lower_bound = 1.4 * med, 0.6 * med
        filtered_id = np.where((self.offset < upper_bound) & (self.offset > lower_bound))[0]
        f_offset = self.offset[filtered_id]
        print("Sigma > " + str(upper_bound) + " nm and sigma < " + str(lower_bound) + " nm filtered")

        plt.hist(f_offset, bins=100, density=1)
        plt.axvline(x=upper_bound, ls="--")
        plt.axvline(x=lower_bound, ls="--")

        plt.xlabel("Offset (photon)")
        plt.title("Offset filtering\n" + self.name + ", bins=100")
        if plt_save:
            plt.savefig(self.path + "/filter_offset.png", dpi=300)
            plt.clf()
            plt.close()
        else:
            # plt.show()
            plt.clf()
            plt.close()

        if update_prop:
            self.filtered_id = np.append(self.filtered_id, filtered_id)

    def data_filtering(self, filter_sigma, filter_offset, filter_intensity, filter_uncertainty):
        val, count = np.unique(self.filtered_id, return_counts=True)

        count_true = np.count_nonzero(np.array([filter_offset, filter_intensity, filter_uncertainty, filter_sigma]))
        self.filtered_id = val[count == count_true].astype(int)
        self.ratio_filter = (self.id.size - self.filtered_id.size) / self.id.size

        print("Final " + str(self.id.size - self.filtered_id.size) + " data points filtered!")

        self.id = self.id[self.filtered_id]
        self.frame = self.frame[self.filtered_id]
        self.x = self.x[self.filtered_id]  # nm
        self.y = self.y[self.filtered_id]  # nm
        self.sigma = self.sigma[self.filtered_id]  # nm
        self.intensity = self.intensity[self.filtered_id]  # photons
        self.offset = self.offset[self.filtered_id]  # photon
        self.bkgstd = self.bkgstd[self.filtered_id]  # photon
        self.uncertainty = self.uncertainty[self.filtered_id]  # nm

    def plot_coord_filtered(self, zoom=5000):
        plt.scatter(self.x, self.y, s=0.5)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.title(self.name + ", filtered, s=1")
        plt.savefig(self.path + "/coords_filtered.png", dpi=300)
        plt.clf()
        plt.close()

        fig, _ = plt.subplots(figsize=(6, 6))
        plt.scatter(self.x, self.y, s=0.5)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.title(self.name + ", filtered, s=1")
        plt.xlim([50000 / 2 - zoom, 50000 / 2 + zoom])
        plt.ylim([60000 / 2 - zoom, 60000 / 2 + zoom])
        plt.title(self.name + ", filtered, zoomed, s=1")
        plt.savefig(self.path + "/coords_filtered_zoomed.png", dpi=300)
        plt.clf()
        plt.close()

    def calc_loc_precision(self):
        self.loc_precision = self.uncertainty.max()
        print("Updated localization precision = " + str(self.loc_precision) + " nm")

    def apply_drift_correction(self):
        drift_x = np.load(self.path_drift + "/x_drift.npy")
        drift_y = np.load(self.path_drift + "/y_drift.npy")

        for frame in tqdm(range(self.n_frame)):
            if frame != 0:
                this_frame = self.frame == frame
                self.x[this_frame] = self.x[this_frame] - drift_x[frame - 1]
                self.y[this_frame] = self.y[this_frame] - drift_y[frame - 1]

    def pre_processing(self, drift_correct, filter_sigma=True, filter_intensity=True, filter_offset=True,
                       filter_uncertainty=True, plt_save=False):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if drift_correct:
            PAINTFiltered.apply_drift_correction(self)

        if filter_offset:
            PAINTFiltered.filter_offset(self, plt_save=plt_save, update_prop=True)
        if filter_uncertainty:
            PAINTFiltered.filter_uncertainty(self, plt_save=plt_save, update_prop=True)
        if filter_sigma:
            PAINTFiltered.filter_sigma(self, plt_save=plt_save, update_prop=True)
        if filter_intensity:
            PAINTFiltered.filter_intensity(self, plt_save=plt_save, update_prop=True)

        PAINTFiltered.data_filtering(self, filter_sigma, filter_offset, filter_intensity, filter_uncertainty)
        if plt_save:
            PAINTFiltered.plot_coord_filtered(self)
        PAINTFiltered.calc_loc_precision(self)

        if not os.path.exists(self.path + "/FilteredData"):
            os.mkdir(self.path + "/FilteredData")

        np.save(self.path + "/FilteredData/frame.npy", self.frame)
        np.save(self.path + "/FilteredData/x.npy", self.x)
        np.save(self.path + "/FilteredData/y.npy", self.y)
        np.save(self.path + "/FilteredData/intensity.npy", self.intensity)
        np.save(self.path + "/FilteredData/loc_precision.npy", self.loc_precision)
        np.save(self.path + "/FilteredData/n_frame.npy", self.n_frame)
        np.save(self.path + "/FilteredData/ratio_filter.npy", self.ratio_filter)


class DriftCorrection(PAINT):

    def __init__(self, name, path):
        PAINT.__init__(self, name, path)
        self.path = path + "/DriftCorrection"

    def filter_sigma(self, plt_save=False):
        med = np.median(self.sigma)
        upper_bound, lower_bound = med * 1.5, med * 0.5
        filtered_id = np.where((self.sigma < upper_bound) & (self.sigma > lower_bound))[0]
        f_sigma = self.sigma[filtered_id]
        print("Sigma > " + str(upper_bound) + " nm and sigma < " + str(lower_bound) + " nm filtered")

        plt.hist(f_sigma, bins=100, density=1)
        plt.axvline(x=upper_bound, ls="--")
        plt.axvline(x=lower_bound, ls="--")
        plt.xlabel("Sigma (nm)")
        plt.title("Sigma filtering\n" + self.name + ", bins=100")
        if plt_save:
            plt.savefig(self.path + "/filter_sigma.png", dpi=300)
            plt.clf()
            plt.close()
        else:
            plt.show()
            plt.clf()
            plt.close()

        # Update self
        print("Final " + str(self.id.size - filtered_id.size) + " data points filtered!")

        self.id = self.id[filtered_id]
        self.frame = self.frame[filtered_id]
        self.x = self.x[filtered_id]  # nm
        self.y = self.y[filtered_id]  # nm
        self.sigma = self.sigma[filtered_id]  # nm
        self.intensity = self.intensity[filtered_id]  # photons
        self.offset = self.offset[filtered_id]  # photon
        self.bkgstd = self.bkgstd[filtered_id]  # photon
        self.uncertainty = self.uncertainty[filtered_id]  # nm

    def mean_shift_clustering(self, bandwidth=200):
        pts = np.stack((self.x, self.y), axis=-1)

        t = time()

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(pts)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        print("Mean shift clustering took %0.2f seconds" % (time() - t))

        n_clusters_ = np.unique(labels).size

        print("number of estimated clusters : %d" % n_clusters_)

        return labels, cluster_centers, pts, bandwidth

    def plot_clustering_result(self, labels, cluster_centers, pts, radius, density_filter=False, threshold=20,
                               plt_save=False):
        fig, ax = plt.subplots()
        colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
        if density_filter:
            val = density_filtering(labels, count_threshold=threshold)
        else:
            val = np.unique(labels)

        for k, col in zip(iter(val), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.scatter(pts[my_members, 0], pts[my_members, 1], s=1, color=col, marker=".")
            plt.plot(
                cluster_center[0],
                cluster_center[1],
                "k+",
                markersize=5,
            )
            draw_circle = plt.Circle(cluster_center, radius, fill=False, lw=1)
            ax.add_artist(draw_circle)

        plt.xlim([0, 49020])
        plt.ylim([0, 58500])
        plt.title("Estimated number of clusters: %d" % val.size)
        if plt_save:
            if density_filter:
                save_name = self.path + "/clustering_density-filtered_" + str(threshold) + ".png"
            else:
                save_name = self.path + "/clustering.png"
            plt.savefig(save_name, dpi=300)
            plt.clf()
            plt.close()
        else:
            plt.show()

        return val

    def calc_drift(self, pts, labels, val):
        drift_x = np.zeros((val.size, self.n_frame - 1))
        drift_y = np.zeros((val.size, self.n_frame - 1))
        x_initial = np.zeros(val.size)
        y_initial = np.zeros(val.size)

        fig, ax = plt.subplots(nrows=2, sharex="all")
        for k in val:
            x = pts[labels == k, 0]
            y = pts[labels == k, 1]
            frame_per_cluster = self.frame[labels == k]
            if frame_per_cluster.size > self.n_frame:
                x_copy = np.zeros(self.n_frame)
                y_copy = np.zeros(self.n_frame)
                v, count = np.unique(frame_per_cluster, return_counts=True)
                more_than_1_count = np.where(count > 1)[0]
                for i in range(self.n_frame):
                    if i in more_than_1_count:
                        x_copy[i] = np.mean(x[frame_per_cluster == i])
                        y_copy[i] = np.mean(y[frame_per_cluster == i])
                    else:
                        x_copy[i] = x[i]
                        y_copy[i] = y[i]
                x = x_copy
                y = y_copy
                frame_per_cluster = v

                drift_x[k, :] = x[1:] - x[0]
                drift_y[k, :] = y[1:] - y[0]
                x_initial[k] = x[0]
                y_initial[k] = y[0]
            elif frame_per_cluster.size < self.n_frame:
                drift_x[k, frame_per_cluster[1:].astype(int) - 1] = x[1:] - x[0]
                drift_y[k, frame_per_cluster[1:].astype(int) - 1] = y[1:] - y[0]
                x_initial[k] = x[0]
                y_initial[k] = y[0]
                frame_per_cluster = np.arange(self.n_frame)
            else:
                drift_x[k, :] = x[1:] - x[0]
                drift_y[k, :] = y[1:] - y[0]
                x_initial[k] = x[0]
                y_initial[k] = y[0]

            # Plot drift
            ax[0].plot(frame_per_cluster[1:], drift_x[k, :])
            ax[1].plot(frame_per_cluster[1:], drift_y[k, :])

        ax[1].set_ylabel("y_drift")
        ax[1].set_xlabel("frame")
        ax[0].set_ylabel("x_drift")
        plt.savefig(self.path + "/drift_per_clust.png", dpi=300)

        return drift_x, drift_y, x_initial, y_initial, ax

    def drift_correct(self, sigma_filter):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if sigma_filter:
            DriftCorrection.filter_sigma(self, plt_save=True)

        labels, cluster_centers, pts, bandwidth = DriftCorrection.mean_shift_clustering(self)

        DriftCorrection.plot_clustering_result(self, labels, cluster_centers, pts, bandwidth, plt_save=True)
        val = DriftCorrection.plot_clustering_result(self, labels, cluster_centers, pts, bandwidth, density_filter=True,
                                                     plt_save=True)

        drift_x, drift_y, x_initial, y_initial, ax = DriftCorrection.calc_drift(self, pts, labels, val)

        ind = []
        for k in range(drift_x.shape[0]):
            if drift_x[k, :][drift_x[k, :] != 0].size < int(0.5 * self.n_frame):
                ind.append(k)
        drift_x = np.delete(drift_x, ind, 0)
        drift_y = np.delete(drift_y, ind, 0)

        print(str(drift_x.shape[0]) + " clusters are used to calculate overall drift in x and y")

        drift_x_final = np.mean(drift_x, axis=0)
        drift_y_final = np.mean(drift_y, axis=0)
        drift_x_std = np.std(drift_x, axis=0)
        drift_y_std = np.std(drift_y, axis=0)
        initial_pos = np.stack((x_initial, y_initial), axis=-1)

        np.save(self.path + "/xy_initial.npy", initial_pos)
        np.save(self.path + "/x_drift.npy", drift_x_final)
        np.save(self.path + "/y_drift.npy", drift_y_final)

        ax[0].clear()
        ax[1].clear()

        ax[0].plot(np.arange(1, self.n_frame), drift_x_std)
        ax[0].set_ylabel("Standard error of x_drift (nm)")
        ax[1].plot(np.arange(1, self.n_frame), drift_y_std)
        ax[1].set_ylabel("Standard error of y_drift (nm)")
        ax[1].set_xlabel("frame")

        plt.savefig(self.path + "/final_drift_std.png", dpi=300)

        ax[0].clear()
        ax[1].clear()

        ax[0].plot(np.arange(1, self.n_frame), drift_x_final)
        ax[0].set_ylabel("x_drift_final (nm)")
        ax[1].plot(np.arange(1, self.n_frame), drift_y_final)
        ax[1].set_ylabel("y_drift_final (nm)")
        ax[1].set_xlabel("frame")

        plt.savefig(self.path + "/final_drift.png", dpi=300)
        plt.clf()
        plt.close()

        for i in val:
            try:
                x = pts[labels == i, 0]
                y = pts[labels == i, 1]
                x_corr = np.append(x[0], x[1:] - drift_x_final)
                y_corr = np.append(y[0], y[1:] - drift_y_final)

                plt.scatter(x, y)
                plt.scatter(x_corr, y_corr)
                plt.title(str(i))
                plt.xlabel("x (nm)")
                plt.ylabel("y (nm)")
                plt.savefig(self.path + "/fiducial_" + str(i) + ".png", dpi=300)
                plt.clf()
            except ValueError:
                pass

        plt.close()


class MergedLoc:
    def __init__(self, folder_list, path, n_blocks, frame_per_block):
        self.folder_list = folder_list
        self.path = path
        self.path_merged = self.path + "/PreProcessing"
        self.n_blocks = n_blocks
        self.frame_per_block = frame_per_block

    def make_merged_folder(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if not os.path.exists(self.path_merged):
            os.mkdir(self.path_merged)

        if not os.path.exists(self.path_merged + "/FilteredData"):
            os.mkdir(self.path_merged + "/FilteredData")

    def drift_between_blocks(self, block_num, x, y):
        fiducial_pos_in_frame_1 = np.load(self.folder_list[0] + "/DriftCorrection/xy_initial.npy")
        fiducial_pos = np.load(self.folder_list[block_num] + "/DriftCorrection/xy_initial.npy")
        drift_x = []
        drift_y = []
        for pos in fiducial_pos:
            dist = []
            for pos_0 in fiducial_pos_in_frame_1:
                dist.append(np.sqrt((pos[0] - pos_0[0]) ** 2 + (pos[1] - pos_0[1]) ** 2))
            if min(dist) < 1000:
                drift_x.append(pos[0] - fiducial_pos_in_frame_1[dist == min(dist)][0][0])
                drift_y.append(pos[1] - fiducial_pos_in_frame_1[dist == min(dist)][0][1])
        drift_x = np.mean(drift_x)
        drift_y = np.mean(drift_y)
        x = x - drift_x
        y = y - drift_y
        print("Drift between blocks have been corrected!")

        return x, y

    def plot_merged_loc(self, x, y, zoom=5000):
        plt.scatter(x, y, s=0.5)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.title(self.path + ", filtered, s=1")
        plt.savefig(self.path_merged + "/coords_filtered.png", dpi=300)
        plt.clf()
        plt.close()

        fig, _ = plt.subplots(figsize=(6, 6))
        plt.scatter(x, y, s=0.5)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.title(self.path + ", filtered, s=1")
        plt.xlim([50000 / 2 - zoom, 50000 / 2 + zoom])
        plt.ylim([60000 / 2 - zoom, 60000 / 2 + zoom])
        plt.title(self.path + ", filtered, zoomed, s=1")
        plt.savefig(self.path_merged + "/coords_filtered_zoomed.png", dpi=300)
        plt.clf()
        plt.close()

    def merge_data(self, drift_correct, total_frames):
        MergedLoc.make_merged_folder(self)

        x_merged, y_merged, frame_merged, intensity_merged, loc_precision_merged = \
            load_data_from_folder(self.folder_list[0])

        for i in np.arange(1, self.n_blocks):
            x, y, frame, intensity, loc_precision = load_data_from_folder(self.folder_list[i])
            if drift_correct:
                x, y = MergedLoc.drift_between_blocks(self, 1, x, y)
            x_merged = np.append(x_merged, x)
            y_merged = np.append(y_merged, y)
            frame_merged = np.append(frame_merged, frame + i * self.frame_per_block)
            intensity_merged = np.append(intensity_merged, intensity)
            loc_precision_merged = np.append(loc_precision_merged, loc_precision)

        print(loc_precision_merged)
        loc_precision_merged = np.max(loc_precision_merged)
        print(loc_precision_merged)
        MergedLoc.plot_merged_loc(self, x_merged, y_merged)

        np.save(self.path_merged + "/FilteredData/frame.npy", frame_merged)
        np.save(self.path_merged + "/FilteredData/x.npy", x_merged)
        np.save(self.path_merged + "/FilteredData/y.npy", y_merged)
        np.save(self.path_merged + "/FilteredData/intensity.npy", intensity_merged)
        np.save(self.path_merged + "/FilteredData/loc_precision.npy", loc_precision_merged)
        if np.unique(frame_merged).size == total_frames:
            np.save(self.path_merged + "/FilteredData/n_frame.npy", np.unique(frame_merged).size)
        else:
            np.save(self.path_merged + "/FilteredData/n_frame.npy", total_frames)


class Clustering:

    def __init__(self, path):
        path_data = path + "/PreProcessing/FilteredData/"
        self.x = np.load(path_data + "x.npy")
        self.y = np.load(path_data + "y.npy")
        self.frame = np.load(path_data + "frame.npy")
        self.intensity = np.load(path_data + "intensity.npy")
        self.n_samples = self.x.size
        self.loc_precision = np.load(path_data + "loc_precision.npy")
        self.path = path + "/Clustering"

    def select_pts_within_region(self, n_x, n_y, roi_size=5000):
        pts = np.stack((self.x, self.y), axis=-1)

        # print("Region of interest %d nm" % roi_size)
        roi_size = roi_size + 500
        # print("Region that will be analyzed %d nm" % roi_size)
        x_min, x_max, y_min, y_max = find_roi(n_x, n_y, roi_size)

        cond_x = (pts[:, 0] <= x_max) & (pts[:, 0] >= x_min)
        cond_y = (pts[:, 1] <= y_max) & (pts[:, 1] >= y_min)

        pts_roi = pts[cond_x & cond_y]
        frame = self.frame[cond_x & cond_y]
        intensity = self.intensity[cond_x & cond_y]

        return pts_roi, frame, intensity

    def mean_shift_clustering(self, select_roi=True, n_x=0.5, n_y=0.5, path_folder=""):
        np.random.seed(1000)
        if select_roi:
            pts, frame, intensity = Clustering.select_pts_within_region(self, n_x, n_y)
        else:
            pts = np.stack((self.x, self.y), axis=-1)
            frame = self.frame
            intensity = self.intensity

        t = time()
        bandwidth = self.loc_precision * 2  # 50
        print("Bandwidth = %d" % bandwidth)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(pts)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        print("Mean shift clustering took %0.2f seconds" % (time() - t))

        n_clusters_ = np.unique(labels).size

        print("number of estimated clusters : %d" % n_clusters_)

        # Plot results
        Clustering.plot_clustering_result(self, labels, cluster_centers, pts, bandwidth, density_filter=False,
                                          plt_save=True, path_folder=path_folder)

        np.save(self.path + path_folder + "/ms_labels.npy", labels)
        np.save(self.path + path_folder + "/ms_clust_centers.npy", cluster_centers)
        np.save(self.path + path_folder + "/ms_frames.npy", frame)
        np.save(self.path + path_folder + "/ms_intensity.npy", intensity)

    def birch_clustering(self, threshold_offset=1):
        pts = np.stack((self.x, self.y), axis=-1)

        t = time()
        threshold = self.loc_precision * 5 * threshold_offset

        birch_mod = Birch(threshold=threshold, n_clusters=None).fit(pts)
        labels = birch_mod.labels_
        cluster_centers = birch_mod.subcluster_centers_
        print("BIRCH clustering took %0.2f seconds" % (time() - t))

        n_clusters_ = np.unique(labels).size
        print("number of estimated clusters : %d" % n_clusters_)

        # Plot results
        Clustering.plot_clustering_result(self, labels, cluster_centers, pts, threshold, density_filter=False,
                                          threshold=10, plt_save=True)
        Clustering.plot_clustering_result(self, labels, cluster_centers, pts, threshold, density_filter=True,
                                          threshold=10, plt_save=True)
        for i in np.arange(5, 100, 10):
            Clustering.plot_clustering_result(self, labels, cluster_centers, pts, threshold, density_filter=True,
                                              threshold=i, plt_save=True)

    def kmeans_clustering(self, n_clust):
        pts = np.stack((self.x, self.y), axis=-1)

        t = time()
        kmean_mod = MiniBatchKMeans(n_clusters=n_clust, verbose=1).fit(pts)
        labels = kmean_mod.labels_
        cluster_centers = kmean_mod.cluster_centers_
        print("Mini batch k-means clustering took %0.2f seconds" % (time() - t))

        n_clusters_ = np.unique(labels).size
        print("number of estimated clusters : %d" % n_clusters_)

        # Plot results
        Clustering.plot_clustering_result(self, labels, cluster_centers, pts, 100, density_filter=False,
                                          threshold=10, plt_save=True)
        Clustering.plot_clustering_result(self, labels, cluster_centers, pts, 100, density_filter=True,
                                          threshold=10, plt_save=True)
        for i in np.arange(5, 100, 10):
            Clustering.plot_clustering_result(self, labels, cluster_centers, pts, 100, density_filter=True,
                                              threshold=i, plt_save=True)

    def dbscan_clustering(self, select_roi=True, n_x=0.5, n_y=0.5):
        if select_roi:
            pts, _, _ = Clustering.select_pts_within_region(self, n_x, n_y)
        else:
            pts = np.stack((self.x, self.y), axis=-1)

        db = DBSCAN(eps=30, min_samples=5).fit(pts)
        X = pts
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                ".",
                c=tuple(col),
                markersize=2,
            )

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                ".",
                c=tuple(col),
                markersize=2,
            )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()

    def plot_clustering_result(self, labels, cluster_centers, pts, radius, density_filter=False, threshold=30,
                               plt_save=False, path_folder=""):
        _, _ = plt.subplots(figsize=(6, 6))
        colors = cycle("bgrcmybgrcmybgrcmybgrcmy")

        if density_filter:
            val = density_filtering(labels, count_threshold=threshold)
        else:
            val = np.unique(labels)

        for k, col in zip(iter(val), colors):
            plot_single_cluster(k, labels, cluster_centers, pts, col, radius)

        # print("Plotting took %0.2f seconds" % (time() - t))

        plt.title("Estimated number of clusters: %d" % val.size)
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        if plt_save:
            if density_filter:
                plt.savefig(self.path + path_folder + "/clustering_density-filtered_" + str(threshold) + ".png",
                            dpi=300)
            else:
                plt.savefig(self.path + path_folder + "/clustering.png", dpi=300)
            plt.clf()
        else:
            plt.show()
        plt.close()

        _, _ = plt.subplots(tight_layout=True)

        n_pts_plot = 15
        if val.size > n_pts_plot:
            chosen_pts = np.random.choice(val, size=n_pts_plot)
        else:
            chosen_pts = val

        for k in chosen_pts:
            x_max, x_min, y_max, y_min = plot_single_cluster(k, labels, cluster_centers, pts, "b", radius)
            # ax.axis("equal")
            plt.ylim([y_min, y_max])
            plt.xlim([x_min, x_max])
            plt.xticks(c="w")
            plt.yticks(c="w")
            plt.title(str(k))
            plt.xlabel("x (nm)")
            plt.ylabel("y (nm)")
            if plt_save:
                if density_filter:
                    plt.savefig(self.path + "/clustering_" + str(k) + "_density-filtered_" + str(threshold) + ".png",
                                dpi=300)
                else:
                    plt.savefig(self.path + path_folder + "/clustering_" + str(k) + ".png", dpi=300)
                plt.clf()
            else:
                plt.show()
        plt.close()

        return val

    def plot_clustering_zoom(self, plt_save=False):
        def select_pts(points, bound):
            x_min_, x_max_, y_min_, y_max_ = bound
            cond_x = (points[:, 0] <= x_max_) & (points[:, 0] >= x_min_)
            cond_y = (points[:, 1] <= y_max_) & (points[:, 1] >= y_min_)
            pts_roi = points[cond_x & cond_y]
            ind_ = np.where(cond_x & cond_y)[0]
            return pts_roi, ind_

        x_percent_roi = [0.4, 0.4, 0.5, 0.6, 0.6]
        y_percent_roi = [0.4, 0.6, 0.5, 0.4, 0.6]
        for n_x, n_y, roi in zip(x_percent_roi, y_percent_roi, np.arange(len(x_percent_roi))):
            path_folder = "/roi" + str(roi)
            pts, _, _ = Clustering.select_pts_within_region(self, n_x, n_y)
            x_min, x_max, y_min, y_max = find_roi(n_x, n_y, 5000)
            labels = np.load(self.path + path_folder + "/ms_labels.npy")
            clust_cents = np.load(self.path + path_folder + "/ms_clust_centers.npy")

            _, index = select_pts(clust_cents, [x_min + 2000, x_max - 2000, y_min + 2000, y_max - 2000])
            val = np.unique(labels)[index]

            _, ax = plt.subplots(figsize=(6, 6))
            colors = cycle("bgrcmybgrcmybgrcmybgrcmy")
            for k, col in zip(iter(val), colors):
                plot_single_cluster(k, labels, clust_cents, pts, col, 50)
                plt.annotate(str(k), clust_cents[k], horizontalalignment='center', verticalalignment='center', size=7,
                             weight='bold', color="k")
            if plt_save:
                plt.savefig(self.path + path_folder + "/clustering_zoom.png", dpi=300)
            else:
                plt.show()
            plt.close()

    def run(self, select_roi=True, multiple_roi=True):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if multiple_roi:
            x_percent_roi = [0.4, 0.4, 0.5, 0.6, 0.6]
            y_percent_roi = [0.4, 0.6, 0.5, 0.4, 0.6]

            for i in range(len(x_percent_roi)):
                path_folder = "/roi" + str(i)
                if not os.path.exists(self.path + path_folder):
                    os.mkdir(self.path + path_folder)
                Clustering.mean_shift_clustering(self, n_x=x_percent_roi[i], n_y=y_percent_roi[i],
                                                 path_folder=path_folder,
                                                 select_roi=select_roi)
            Clustering.plot_clustering_zoom(self, plt_save=True)
        else:
            path_folder = ""
            Clustering.mean_shift_clustering(self, path_folder=path_folder, select_roi=select_roi)


class LifetimeAnalysis:

    def __init__(self, path, frame_rate):
        self.clust_path = path + "/Clustering"
        self.labels = []
        self.frame = []
        self.n_frame = np.load(path + "/PreProcessing/FilteredData/n_frame.npy")
        self.intensity = []
        self.path = path + "/LifetimeAnalysis"
        self.framerate = frame_rate

    def compute_traces_per_clust(self, i):
        frames = np.arange(0, self.n_frame) / self.framerate
        _, _ = plt.subplots(figsize=(8, 5))

        signal = np.zeros(self.n_frame, dtype=np.int64)

        find_frames = self.frame[self.labels == i].astype(int)
        find_intensity = self.intensity[self.labels == i]
        print(find_frames)

        _, count = np.unique(find_frames, return_counts=True)
        if np.any(count != 1):
            print("Cluster label %d has repeated frames!" % i)

        signal[find_frames] = find_intensity

        # Merge events that have less than 1 or 2 frames of off time between on time
        signal = merge_event(find_frames, find_intensity, signal, 0, 2)
        signal = merge_event(find_frames, find_intensity, signal, 0, 3)

        plt.plot(frames, signal, "-")
        plt.xlabel("t (s)")
        plt.ylabel("Intensity (photon)")
        plt.show()

        return signal

    def compute_traces(self, path_folder):
        frames = np.arange(0, self.n_frame) / self.framerate
        _, _ = plt.subplots(figsize=(8, 5))

        v = np.unique(self.labels)
        v_plot = np.random.choice(v, size=10)

        signal = np.zeros((v.size, self.n_frame), dtype=np.int64)
        trace_w_repeated_frame = np.array([], dtype=int)
        k = 0
        for i in v:
            # print(i)
            find_frames = self.frame[self.labels == i].astype(int)
            find_intensity = self.intensity[self.labels == i]

            fr, count = np.unique(find_frames, return_counts=True)
            if np.any(count != 1):
                print("Cluster label %d has repeated frames!" % i)
                trace_w_repeated_frame = np.append(trace_w_repeated_frame, k)
                for f in fr[count == 1]:
                    signal[k, f] = find_intensity[find_frames == f]
                for f in fr[count != 1]:
                    signal[k, f] = np.sum(find_intensity[find_frames == f])
                k += 1
                continue

            signal[k, find_frames] = find_intensity

            # Merge events that have less than 1 or 2 frames of dark time between bright time
            signal = merge_event(find_frames, find_intensity, signal, k, 2)
            # signal = merge_event(find_frames, find_intensity, signal, k, 3)

            if i in v_plot:
                plt.plot(frames, signal[k, :], "-")
                plt.xlabel("t (s)")
                plt.ylabel("Intensity (photon)")
                plt.savefig(self.path + path_folder + "/trace_" + str(i) + ".png", dpi=300)
                plt.clf()

            k += 1
        plt.close()

        # signal = np.delete(signal, trace_w_repeated_frame, axis=0)
        np.save(self.path + path_folder + "/signal.npy", signal)
        np.save(self.path + path_folder + "/trace_w_repeated_frames", trace_w_repeated_frame)

        return signal, trace_w_repeated_frame

    def obtain_lifetime_from_all_cluster(self, signal, exclude_censored_lifetimes, path_folder, trace_w_repeated_frame):
        signal[signal != 0] = 1
        dark_lifetimes = np.array([])
        bright_lifetimes = np.array([])
        clust_id_to_filter = np.array([])
        for i in range(signal.shape[0]):
            if i not in trace_w_repeated_frame:
                dark_t, bright_t, _ = compute_lifetime(signal[i, :], exclude_censored_lifetimes)
                dark_lifetimes = np.append(dark_lifetimes, dark_t)
                bright_lifetimes = np.append(bright_lifetimes, bright_t)
                # Label clust when there is only one bright frame & remove single bright frame & check if there are
                # multiple bright frames within traces, if not remove clust in total
                if find_clust_with_n_bright_time(bright_t, 1):
                    clust_id_to_filter = np.append(clust_id_to_filter, i)

        find_short_event = np.where(dark_lifetimes == 1)[0]
        dark_lifetimes = np.delete(dark_lifetimes, find_short_event)
        find_short_event = np.where(bright_lifetimes == 1)[0]
        bright_lifetimes = np.delete(bright_lifetimes, find_short_event)

        np.save(self.path + path_folder + "/clust_id_to_filter.npy", clust_id_to_filter)
        np.save(self.path + path_folder + "/all_dark_times.npy", dark_lifetimes)
        np.save(self.path + path_folder + "/all_bright_times.npy", bright_lifetimes)

        return dark_lifetimes, bright_lifetimes

    def plot_and_save_lifetime_single_expfit(self, dark_lt, lifetimes_nframes, framerate, save, path_folder):
        try:
            if dark_lt:
                x_label = r'$\tau_{off}$' ' (s)'
                save_name = "dark_time"
                lower_ylim = 0.01
            else:
                x_label = r'$\tau_{on}$' ' (s)'
                save_name = "bright_time"
                lower_ylim = 0.0001

            tau, f = ecdf(lifetimes_nframes / framerate)
            param_bounds = ([0, 0], [1, np.inf])
            pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
            tau_bound1 = 1 / pred[0][1]
            # plot
            plt.scatter(tau, 1 - f, lw=2, label='Empirical CDF', color='red', s=1)
            plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]))
            plt.ylim([lower_ylim, 1])
            plt.xlabel(x_label)
            plt.ylabel('1 - cdf')
            plt.yscale('log')
            plt.legend(fancybox=True, loc='upper right')
            plt.title(r'$\tau$' '=  %1.3f' % tau_bound1)

            if save:
                plt.savefig(self.path + path_folder + "/" + save_name + "_single_exp.png", dpi=300)
                plt.clf()
                plt.close()
            else:
                plt.show()
            return tau_bound1
        except ValueError:
            pass

    def plot_and_save_lifetime_double_expfit(self, dark_lt, lifetimes_nframes, framerate, save, path_folder):
        try:
            if dark_lt:
                x_label = r'$\tau_{off}$' ' (s)'
                save_name = "dark_time"
                # lower_ylim = 0.01
            else:
                x_label = r'$\tau_{on}$' ' (s)'
                save_name = "bright_time"
                # lower_ylim = 0.0001

            tau, f = ecdf(lifetimes_nframes / framerate)
            param_bounds = ([0, 0, 0, 0], [1, np.inf, 1, np.inf])
            pred = scipy.optimize.curve_fit(exp_lifetimes2, tau, 1 - f, np.array([0.1, 0.1, 0.1, 0.001]),
                                            bounds=param_bounds)
            tau_1 = 1 / pred[0][1]
            tau_2 = 1 / pred[0][3]
            fract1 = pred[0][0]
            fract2 = 1 - pred[0][0] - pred[0][2]
            # plot
            plt.scatter(tau, 1 - f, lw=2, label='Empirical CDF', color='red', s=1)
            plt.plot(tau, exp_lifetimes2(tau, pred[0][0], pred[0][1], pred[0][2], pred[0][3]))
            # plt.ylim([lower_ylim, 1])
            plt.xlabel(x_label)
            plt.ylabel('1 - cdf')
            plt.yscale('log')
            plt.legend(fancybox=True, loc='upper right')
            plt.title(r'$\tau_1$' '=  %1.3f' % tau_1 + ' , ' + r'$\tau_2$' '=  %1.3f' % tau_2 +
                      '\n fraction1 = %1.3f' % fract1 + ', fraction2 = %1.3f' % fract2)

            if save:
                plt.savefig(self.path + path_folder + "/" + save_name + "_double_exp.png", dpi=300)
                plt.clf()
                plt.close()
            else:
                plt.show()
            return [tau_1, fract1, tau_2, fract2]
        except ValueError:
            pass

    def lifetime(self, multiple_roi=True):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if multiple_roi:
            for i in range(5):
                path_clust_info = self.clust_path + "/roi" + str(i) + "/"
                self.labels = np.load(path_clust_info + "ms_labels.npy")
                self.frame = np.load(path_clust_info + "ms_frames.npy").astype(int)
                self.intensity = np.load(path_clust_info + "ms_intensity.npy")
                folder_path = "/roi" + str(i)
                if not os.path.exists(self.path + folder_path):
                    os.mkdir(self.path + folder_path)

                signal, trace_w_repeated_frame = LifetimeAnalysis.compute_traces(self, folder_path)
                # signal = np.load(self.path + "/signal.npy")
                dark_lifetime, bright_lifetime = LifetimeAnalysis.obtain_lifetime_from_all_cluster(self, signal, True,
                                                                                                   folder_path,
                                                                                                   trace_w_repeated_frame)
                LifetimeAnalysis.plot_and_save_lifetime_single_expfit(self, True, dark_lifetime, self.framerate, True,
                                                                      folder_path)
                LifetimeAnalysis.plot_and_save_lifetime_single_expfit(self, False, bright_lifetime, self.framerate,
                                                                      True,
                                                                      folder_path)
                LifetimeAnalysis.plot_and_save_lifetime_double_expfit(self, False, bright_lifetime, self.framerate,
                                                                      True,
                                                                      folder_path)
        else:
            path_clust_info = self.clust_path + "/"
            self.labels = np.load(path_clust_info + "ms_labels.npy")
            self.frame = np.load(path_clust_info + "ms_frames.npy").astype(int)
            self.intensity = np.load(path_clust_info + "ms_intensity.npy")
            path_folder = ""

            signal, trace_w_repeated_frame = LifetimeAnalysis.compute_traces(self, path_folder)
            # signal = np.load(self.path + "/signal.npy")
            dark_lifetime, bright_lifetime = LifetimeAnalysis.obtain_lifetime_from_all_cluster(self, signal, True,
                                                                                               path_folder,
                                                                                               trace_w_repeated_frame)
            LifetimeAnalysis.plot_and_save_lifetime_single_expfit(self, True, dark_lifetime, self.framerate, True,
                                                                  path_folder)
            LifetimeAnalysis.plot_and_save_lifetime_single_expfit(self, False, bright_lifetime, self.framerate, True,
                                                                  path_folder)
            LifetimeAnalysis.plot_and_save_lifetime_double_expfit(self, False, bright_lifetime, self.framerate, True,
                                                                  path_folder)


class MoleculeDensity:
    def __init__(self, path):
        # Output to final result csv/xlsx
        self.path_clust = path + "/Clustering"
        self.path_la = path + "/LifetimeAnalysis"
        self.path = path + "/MoleculeDensity"

    def determine_density(self, clust_cent, clust_to_filter, roi_size=5000, choose_roi=True, n_x=0.5, n_y=0.5, i="",
                          save_name=""):
        if clust_to_filter is not None:
            clust_cent = np.delete(clust_cent, clust_to_filter.astype(int), 0)
        if choose_roi:
            x_min, x_max, y_min, y_max = find_roi(n_x, n_y)
            area = (roi_size / 1000) ** 2
        else:
            x_min, x_max, y_min, y_max = 0, 50000, 0, 60000
            area = 50 * 60

        cond_x = (clust_cent[:, 0] <= x_max) & (clust_cent[:, 0] >= x_min)
        cond_y = (clust_cent[:, 1] <= y_max) & (clust_cent[:, 1] >= y_min)

        clust_cent_roi = clust_cent[cond_x & cond_y]
        density = clust_cent_roi.shape[0] / area

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(clust_cent_roi[:, 0], clust_cent_roi[:, 1], s=1)
        ax.set_title("Binder density = " + str(density))
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        plt.axis("equal")
        plt.savefig(self.path + "/clust_cent_plot" + i + save_name + ".png", dpi=300)
        plt.clf()
        plt.close()

        np.save(self.path + "/final_clust_center" + i + save_name + ".npy", clust_cent)

        return density

    def run(self, multiple_roi=True, roi_size=5000, choose_roi=True):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if multiple_roi:
            x_percent_roi = [0.4, 0.4, 0.5, 0.6, 0.6]
            y_percent_roi = [0.4, 0.6, 0.5, 0.4, 0.6]

            density_list_unfiltered = np.array([])
            density_list = np.array([])
            density_filter_repeatedframeinclust = np.array([])
            for i in range(len(x_percent_roi)):
                path_folder = "/roi" + str(i)
                clust_cent = np.load(self.path_clust + path_folder + "/ms_clust_centers.npy")
                clust_to_filter = np.load(self.path_la + path_folder + "/clust_id_to_filter.npy")
                clust_w_repeated_frame = np.load(self.path_la + path_folder + "/trace_w_repeated_frames.npy")

                # Unfiltered
                density_unfiltered = MoleculeDensity.determine_density(self, clust_cent, None, n_x=x_percent_roi[i],
                                                                       n_y=y_percent_roi[i], i=str(i),
                                                                       roi_size=roi_size,
                                                                       save_name="_unfiltered")
                density_list_unfiltered = np.append(density_list_unfiltered, density_unfiltered)

                # Filter clusters with only single-frame events
                density = MoleculeDensity.determine_density(self, clust_cent, clust_to_filter, n_x=x_percent_roi[i],
                                                            n_y=y_percent_roi[i], i=str(i), roi_size=roi_size)
                density_list = np.append(density_list, density)

                # Filter clusters with only single-frame events and clusters with multiple events at a time
                filter_id = np.unique(np.append(clust_to_filter, clust_w_repeated_frame))
                density_repeatedframe = MoleculeDensity.determine_density(self, clust_cent, filter_id,
                                                                          n_x=x_percent_roi[i], n_y=y_percent_roi[i],
                                                                          i=str(i), roi_size=roi_size,
                                                                          save_name="_filter-repeated-frames")
                density_filter_repeatedframeinclust = np.append(density_filter_repeatedframeinclust,
                                                                density_repeatedframe)
            np.save(self.path + "/density_multiple_roi_unfiltered.npy", density_list_unfiltered)
            np.save(self.path + "/density_multiple_roi.npy", density_list)
            np.save(self.path + "/density_multiple_roi_filter-repeated-frames.npy", density_filter_repeatedframeinclust)
        else:
            clust_cent = np.load(self.path_clust + "/ms_clust_centers.npy")
            clust_to_filter = np.load(self.path_la + "/clust_id_to_filter.npy")
            clust_w_repeated_frame = np.load(self.path_la + "/trace_w_repeated_frames.npy")

            # Unfiltered
            density_unfiltered = MoleculeDensity.determine_density(self, clust_cent, None, choose_roi=choose_roi,
                                                                   save_name="_unfiltered")
            np.save(self.path + "/density_unfiltered.npy", density_unfiltered)

            # Filter clusters with only single-frame events
            density = MoleculeDensity.determine_density(self, clust_cent, clust_to_filter, choose_roi=choose_roi)
            np.save(self.path + "/density.npy", density)

            # Filter clusters with only single-frame events and clusters with multiple events at a time
            filter_id = np.unique(np.append(clust_to_filter, clust_w_repeated_frame))
            density_repeatedframe = MoleculeDensity.determine_density(self, clust_cent, filter_id,
                                                                      choose_roi=choose_roi,
                                                                      save_name="_filter-repeated-frames")
            np.save(self.path + "/density_filter_filter-repeated-frames.npy", density_repeatedframe)


class ClarkEvansTest:
    def __init__(self, path):
        # Code for CE test
        self.path = path + "/CETest"
        self.path_data = path + "/MoleculeDensity"

    def run(self, roi_size=5000, multiple_roi=True, choose_roi=True):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if multiple_roi:
            x_percent_roi = [0.4, 0.4, 0.5, 0.6, 0.6]
            y_percent_roi = [0.4, 0.6, 0.5, 0.4, 0.6]

            z_score_unfiltered = np.array([])
            z_score = np.array([])
            z_score_filter_repeatedframe = np.array([])
            for i in range(len(x_percent_roi)):
                clust_cent_unfiltered = np.load(self.path_data + "/final_clust_center" + str(i) + "_unfiltered.npy")
                clust_cent = np.load(self.path_data + "/final_clust_center" + str(i) + ".npy")
                clust_cent_filter_repeatedframe = np.load(
                    self.path_data + "/final_clust_center" + str(i) + "_filter-repeated-frames.npy")

                z_score_unfiltered = np.append(z_score_unfiltered, clark_evans_test(clust_cent_unfiltered,
                                                                                    n_x=x_percent_roi[i],
                                                                                    n_y=y_percent_roi[i],
                                                                                    roi_size=roi_size))
                z_score = np.append(z_score, clark_evans_test(clust_cent, n_x=x_percent_roi[i], n_y=y_percent_roi[i],
                                                              roi_size=roi_size))
                z_score_filter_repeatedframe = np.append(z_score_filter_repeatedframe,
                                                         clark_evans_test(clust_cent_filter_repeatedframe,
                                                                          n_x=x_percent_roi[i], n_y=y_percent_roi[i],
                                                                          roi_size=roi_size))
            print(z_score_unfiltered, z_score, z_score_filter_repeatedframe)
            np.save(self.path + "/z_score_multiple_roi_unfiltered.npy", z_score_unfiltered)
            np.save(self.path + "/z_score_multiple_roi.npy", z_score)
            np.save(self.path + "/z_score_multiple_roi_filter-repeated-frames.npy", z_score_filter_repeatedframe)
        else:
            clust_cent_unfiltered = np.load(self.path_data + "/final_clust_center_unfiltered.npy")
            clust_cent = np.load(self.path_data + "/final_clust_center.npy")
            clust_cent_filter_repeatedframe = np.load(self.path_data + "/final_clust_center_filter-repeated-frames.npy")

            z_score_unfiltered = clark_evans_test(clust_cent_unfiltered, roi_size=roi_size, choose_roi=choose_roi)
            z_score = clark_evans_test(clust_cent, roi_size=roi_size, choose_roi=choose_roi)
            z_score_filter_repeatedframe = clark_evans_test(clust_cent_filter_repeatedframe, roi_size=roi_size,
                                                            choose_roi=choose_roi)
            print(z_score_unfiltered, z_score, z_score_filter_repeatedframe)
            np.save(self.path + "/z_score_unfiltered.npy", z_score_unfiltered)
            np.save(self.path + "/z_score.npy", z_score)
            np.save(self.path + "/z_score_filter-repeated-frames.npy", z_score_filter_repeatedframe)


class IntensityCheck:
    def __init__(self, path, frame_rate):
        self.sample = path.split("_")[0] + "_"
        self.clust_path = path + "/Clustering"
        self.path_la = path + "/LifetimeAnalysis"
        self.path_preprocess = path + "/PreProcessing/FilteredData/"
        self.path = path + "/IntensityCheck"
        self.framerate = frame_rate

    def obtain_raw_frame_intensity(self, total_time=20):
        if not os.path.exists(self.path + "/RawData"):
            os.mkdir(self.path + "/RawData")
        frame_per_block = 5 * 60 * self.framerate
        initial_time = np.arange(1, total_time * 60 * self.framerate, frame_per_block)
        final_time = initial_time + frame_per_block - 1
        time_range = ["_" + str(initial_time[i]) + "-" + str(final_time[i]) for i in range(initial_time.size)]

        frames = np.array([])
        intensity = np.array([])

        for time_ in time_range:
            name = self.sample + "red" + time_
            data = pd.read_csv(name + ".csv").values
            frames = np.append(frames, data[:, 1] - 1)
            intensity = np.append(intensity, data[:, 5])

        np.save(self.path + "/RawData/raw_frames.npy", frames)
        np.save(self.path + "/RawData/raw_intensity.npy", intensity)

    def plot_locs_per_frame(self, raw=False, plot=True):
        if raw:
            frame = np.load(self.path + "/RawData/raw_frames.npy")
        else:
            frame = np.load(self.path_preprocess + "frame.npy")
        val, count = np.unique(frame, return_counts=True)
        mean, std = np.mean(count, dtype=int), np.std(count, dtype=int)
        print(np.min(count), mean, np.max(count))
        if plot:
            plt.plot(val, count)
            plt.axhline(y=mean, c="r", ls="-")
            plt.axhline(y=mean - std, c="r", ls="--")
            plt.axhline(y=mean + std, c="r", ls="--")
            plt.title(self.sample.split("_")[0])
            plt.ylim(bottom=0)
            # plt.hist(frame, bins=100)
            plt.show()

    def n_events_per_clust(self, exclude_censored_lifetimes=True, plot=False):
        n_events = np.array([])
        for k in range(5):
            subfolder = "roi" + str(k)
            signal = np.load(self.path_la + "/" + subfolder + "/signal.npy")
            trace_w_repeated_frame = np.load(self.path_la + "/" + subfolder + "/trace_w_repeated_frames.npy")
            signal[signal != 0] = 1
            n_events_per_clust = np.zeros(signal.shape[0], dtype=np.int8)
            for i in range(signal.shape[0]):
                if i not in trace_w_repeated_frame:
                    _, bright_t, _ = compute_lifetime(signal[i, :], exclude_censored_lifetimes)
                    n_events_per_clust[i] = bright_t.size
            n_events = np.append(n_events, n_events_per_clust)
        if plot:
            plt.hist(n_events, bins=100)
            plt.xlabel("No. events")
            plt.ylabel("Count")
            plt.xlim(left=0)
            plt.show()
        return n_events

    def find_intensity(self, subfolder="", exclude_censored_lifetimes=True, plot=False):
        signal = np.load(self.path_la + "/" + subfolder + "/signal.npy")
        trace_w_repeated_frame = np.load(self.path_la + "/" + subfolder + "/trace_w_repeated_frames.npy")
        intensity_single_frame = np.array([])
        other_intensity = np.array([])
        for i in range(signal.shape[0]):
            if i not in trace_w_repeated_frame:
                _, bright_t, intensity = compute_lifetime(signal[i, :], exclude_censored_lifetimes)
                single_frame_events = np.where(bright_t == 1)[0]
                if single_frame_events.size != 0:
                    for k in single_frame_events:
                        intensity_single_frame = np.append(intensity_single_frame, intensity[k])
                    for k in np.where(bright_t != 1)[0]:
                        other_intensity = np.append(other_intensity, intensity[k])
                else:
                    for k in range(len(intensity)):
                        other_intensity = np.append(other_intensity, intensity[k])

        if plot:
            fig, ax = plt.subplots(1, 2, tight_layout=True, sharex='all', sharey='all')

            ax[0].hist(intensity_single_frame, bins=250)
            ax[0].set_title("Single-frame events")

            ax[1].hist(other_intensity, bins=250)
            ax[1].set_title("Filtered events")

            fig.supylabel("Count")
            fig.supxlabel("Intensity (photons)")
            plt.show()

        return intensity_single_frame, other_intensity

    def obtain_lifetime_from_all_cluster(self, subfolder="", exclude_censored_lifetimes=True):
        signal = np.load(self.path_la + "/" + subfolder + "/signal.npy")
        trace_w_repeated_frame = np.load(self.path_la + "/" + subfolder + "/trace_w_repeated_frames.npy")
        signal[signal != 0] = 1
        dark_lifetimes = np.array([])
        bright_lifetimes = np.array([])
        for i in range(signal.shape[0]):
            if i not in trace_w_repeated_frame:
                dark_t, bright_t, _ = compute_lifetime(signal[i, :], exclude_censored_lifetimes)
                dark_lifetimes = np.append(dark_lifetimes, dark_t)
                bright_lifetimes = np.append(bright_lifetimes, bright_t)

        np.save(self.path + "/" + subfolder + "_unboundLT.npy", dark_lifetimes)
        np.save(self.path + "/" + subfolder + "_boundLT.npy", bright_lifetimes)

        return dark_lifetimes, bright_lifetimes

    def plot_lifetime(self, lifetimes_nframes, unbound_LT, single_exp_fit):
        if unbound_LT:
            x_label = "Unbound state lifetimes (s)"
        else:
            x_label = "Bound state lifetimes (s)"

        lifetimes = lifetimes_nframes / self.framerate
        # lifetimes = lifetimes[lifetimes > 3]
        tau, f = ecdf(lifetimes)
        plt.scatter(tau, 1 - f, lw=2, label='Empirical CDF', color='red', s=1)

        if single_exp_fit:
            param_bounds = ([0, 0], [1, np.inf])
            pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
            tau_bound1 = 1 / pred[0][1]
            plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]))
            print(r'$\tau$' '=  %1.3f' % tau_bound1)
        else:
            param_bounds = ([0, 0, 0, 0, 0, 0], [1, np.inf, 1, np.inf, 1, np.inf])
            pred = scipy.optimize.curve_fit(exp_lifetimes3, tau, 1 - f, np.array([0.1, 0.1, 0.1, 0.001, 0.1, 0.001]),
                                            bounds=param_bounds)
            tau_1 = 1 / pred[0][1]
            tau_2 = 1 / pred[0][3]
            fract1 = pred[0][0]
            fract2 = 1 - pred[0][0] - pred[0][2]
            tau_3 = 1 / pred[0][5]
            fract3 = 1 - pred[0][0] - pred[0][2] - pred[0][4]
            plt.plot(tau, exp_lifetimes3(tau, pred[0][0], pred[0][1], pred[0][2], pred[0][3], pred[0][4], pred[0][5]))
            print(
                r'$\tau_1$' '=  %1.3f' % tau_1 + ' , ' + r'$\tau_2$' '=  %1.3f' % tau_2 + r'$\tau_3$' '=  %1.3f' % tau_3 +
                '\n fraction1 = %1.3f' % fract1 + ', fraction2 = %1.3f' % fract2 + ', fraction3 = %1.3f' % fract3)  # + ', fraction2 = %1.3f' % fract2

        plt.ylim([0.01, 1])
        plt.xlabel(x_label)
        plt.ylabel('1 - cdf')
        plt.yscale('log')
        plt.legend(fancybox=True, loc='upper right')
        plt.show()

    def plot_intensity(self, raw=False, plt_save=True, inv_thresh=2e-8, multiple_roi=True):
        def find_inversion_pts(n_, bin_, plot_=False, inversion_threshold=2e-8):
            intensity_hist = (bin_[:-1] + bin_[1:]) / 2
            first_der = np.diff(n_) / np.diff(intensity_hist)
            signs = np.sign(first_der)
            sign_change = np.where(signs[:-1] != signs[1:])[0]
            min_pts = np.array([], dtype=np.int)
            max_pts = np.array([], dtype=np.int)
            for s in sign_change:
                if (np.abs(first_der[s]) > inversion_threshold) & (np.abs(first_der[s + 1]) > inversion_threshold):
                    if (signs[s] == 1) and (signs[s + 1] == -1):
                        max_pts = np.append(max_pts, s)
                    else:
                        min_pts = np.append(min_pts, s)
            if plot_:
                plt.plot(intensity_hist, n)
                plt.plot(intensity_hist[min_pts + 1], n[min_pts + 1], "k.")
                plt.plot(intensity_hist[max_pts + 1], n[max_pts + 1], "b.")

            return min_pts + 1, max_pts + 1, intensity_hist

        if raw:
            intensity = np.load(self.path + "/RawData/raw_intensity.npy")
            intensity = intensity[intensity < 5000]
        else:
            if multiple_roi:
                intensity = np.array([])
                for i in range(5):
                    signal = np.load(self.path_la + "/roi" + str(i) + "/signal.npy")
                    intensity = np.append(intensity, signal[np.nonzero(signal)])
            else:
                signal = np.load(self.path_la + "/signal.npy")
                intensity = signal[np.nonzero(signal)]
        n, bins, _ = plt.hist(intensity, bins=100, density=True)
        mins, maxs, x = find_inversion_pts(n, bins, inversion_threshold=inv_thresh)
        print(x[mins])
        print(x[maxs])
        # for min_ in mins:
        #     plt.axvline(x[min_], ls="--", c="r")
        # plt.plot(x[maxs], n[maxs], "k.")
        plt.xlabel("Intensity (photons)")
        plt.title(self.sample.split("_")[0])
        plt.ylabel("Normalized count")
        plt.xlim(right=10000)
        if plt_save:
            plt.savefig(self.path + "/intensity_hist.png", dpi=300)
        else:
            plt.show()
        plt.clf()
        np.save(self.path + "/intensity_threshold.npy", x[mins])

    def classify_intensity_in_cluster(self):
        threshold = np.load(self.path + "/intensity_threshold.npy")
        for i in range(5):
            signal = np.load(self.path_la + "/roi" + str(i) + "/signal.npy")
            multi_count = np.array([], dtype=np.int)
            for clust in range(signal.shape[0]):
                single_data = signal[clust, :]
                intensity = single_data[single_data > 0]
                if np.any(intensity > threshold):
                    multi_count = np.append(multi_count, clust)
            np.save(self.path + "/roi" + str(i) + "_ClustMultiCount.npy", multi_count)

    def run(self, multiple_roi=True):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        all_bright_LT = np.array([])
        all_dark_LT = np.array([])
        if multiple_roi:
            for i in range(5):
                sub_folder = "roi" + str(i)
                dark_LT, bright_LT = IntensityCheck.obtain_lifetime_from_all_cluster(self, subfolder=sub_folder)
                all_bright_LT = np.append(all_bright_LT, bright_LT)
                all_dark_LT = np.append(all_dark_LT, dark_LT)
        else:
            sub_folder = ""
            dark_LT, bright_LT = IntensityCheck.obtain_lifetime_from_all_cluster(self, subfolder=sub_folder)
            all_bright_LT = np.append(all_bright_LT, bright_LT)
            all_dark_LT = np.append(all_dark_LT, dark_LT)
        np.save(self.path + "/all_unboundLT.npy", all_dark_LT)
        np.save(self.path + "/all_boundLT.npy", all_bright_LT)

        # all_bright_LT = np.load(self.path + "/all_boundLT.npy")
        # IntensityCheck.plot_lifetime(self, all_bright_LT, False, False)
        # n_events = IntensityCheck.n_events_per_clust(self, signal, plot=False)


class qPAINT(Clustering):

    def __init__(self, path, framerate, path_lists=None):
        Clustering.__init__(self, path)
        self.frame_rate = framerate
        self.n_frame = np.load(path + "/PreProcessing/FilteredData/n_frame.npy")
        self.save_path = path + "/qPAINT"
        self.path_lists = path_lists

    def td_tb_ratio_check(self, area_size, n_x, n_y, exclude_censored_lifetimes, trials, roi_id, save=False, plot_trace=False):

        def select_pts(pts, frame, bound):
            x_min, x_max, y_min, y_max = bound
            cond_x = (pts[:, 0] <= x_max) & (pts[:, 0] >= x_min)
            cond_y = (pts[:, 1] <= y_max) & (pts[:, 1] >= y_min)
            pts_roi = pts[cond_x & cond_y]
            frame = frame[cond_x & cond_y]

            return pts_roi, frame

        def plot_traces(n_frame, single_signal, trial):
            frame_num = np.arange(0, n_frame, 1)
            plt.plot(frame_num, single_signal)
            plt.xlabel("Frame")
            plt.ylabel("Signal")
            plt.yticks(np.arange(2))
            plt.savefig(self.save_path + "/precision_optimisation/" + str(trial) + "_traces0.png", dpi=300)
            plt.clf()
            # plt.show()

        result = []

        raw_pts = np.stack((self.x, self.y), axis=-1)
        bounds = find_roi(n_x, n_y, 5000)
        pts_in_roi, frames = select_pts(raw_pts, self.frame, bounds)

        x_grid = np.arange(bounds[0], bounds[1] + 1, area_size)
        y_grid = np.arange(bounds[2], bounds[3] + 1, area_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Randomly choose 10 area for analysis
        sampling_array = np.arange(0, xx.shape[0] * (xx.shape[1] - 1), 1)
        sampling_array = np.setdiff1d(sampling_array, np.arange(x_grid.size - 1, np.max(sampling_array) + 1, x_grid.size))
        try:
            sampling_index = np.random.choice(sampling_array, 10, replace=False)
        except ValueError:
            sampling_index = sampling_array

        # Compute traces from areas chosen
        xx_flat, yy_flat = xx.flatten(), yy.flatten()
        signal = np.zeros((10, self.n_frame), dtype=np.int64)
        ub_lt = np.array([])
        b_lt = np.array([])
        loc_nums = np.array([])
        double_locs = False
        k = 0
        for index in sampling_index:
            bounding_box = [xx_flat[index], xx_flat[index + 1], yy_flat[index], yy_flat[index + x_grid.size]]
            pts_box, frame_box = select_pts(pts_in_roi, frames, bounding_box)
            signal[k, frame_box.astype(int)] = 1

            _, count = np.unique(frame_box, return_counts=True)
            ratio = count[count != 1].size / count.size
            if np.any(count != 1) and ratio > 0.1:
                double_locs = True

            # Merge events that have less than 1 or 2 frames of dark time between bright time
            signal = merge_event(frame_box.astype(int), None, signal, k, 2)
            # signal = merge_event(frame_box.astype(int), None, signal, k, 3)
            dark_t, bright_t, _ = compute_lifetime(signal[k, :], exclude_censored_lifetimes)
            loc_nums = np.append(loc_nums, dark_t.size + bright_t.size)
            ub_lt = np.append(ub_lt, dark_t)
            b_lt = np.append(b_lt, bright_t)
            k += 1

        if np.any(loc_nums < 100):
            result.append(False)
            raise ValueError("Insufficient number of events (tau_b + tau_d). Minimum 100 events is needed.")
        else:
            result.append(True)

        if double_locs:
            raise AttributeError("Double events detected in selected area! Choose a smaller area of interest.")

        if plot_trace:
            plot_traces(self.n_frame, signal[0, :], trials)

        lifetime = [ub_lt, b_lt]
        labels = ["Unbound state lifetimes", "Bound state lifetimes"]
        col = ["blue", "red"]
        for j in range(2):
            tau, f = ecdf(lifetime[j] / self.frame_rate)
            plt.scatter(tau, 1 - f, lw=2, label=labels[j], color=col[j], s=1)

            param_bounds = ([0, 0], [1, np.inf])
            pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
            tau_bound1 = 1 / pred[0][1]
            # print(tau_bound1)
            result.append(tau_bound1)
            plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]), color=col[j], ls="--")
        plt.ylim([0.01, 1])
        plt.xlabel("State lifetimes (s)")
        plt.ylabel('1 - cdf')
        plt.yscale('log')
        plt.legend(fancybox=True, loc='upper right')
        if save:
            plt.savefig(self.save_path + "/precision_optimisation/lifetime_" + str(trials) + "_" + str(roi_id) + ".png",
                        dpi=300)
        plt.clf()

        result.append(result[1] / result[2])
        return result, lifetime

    def run_optimisation(self, multiple_roi, initial_val, step_size):

        def plot_lifetime(ub_lifetimes, b_lifetimes, plot, trials):
            lifetime = [ub_lifetimes, b_lifetimes]
            labels = ["Unbound state lifetimes", "Bound state lifetimes"]
            col = ["blue", "red"]
            tau_bound = []
            for j in range(2):
                tau, f = ecdf(lifetime[j] / self.frame_rate)
                plt.scatter(tau, 1 - f, lw=2, label=labels[j], color=col[j], s=1)

                param_bounds = ([0, 0], [1, np.inf])
                pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
                tau_bound.append(1 / pred[0][1])
                # print(tau_bound1)
                plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]), color=col[j], ls="--")
            plt.ylim([0.01, 1])
            plt.xlabel("State lifetimes (s)")
            plt.ylabel('1 - cdf')
            plt.yscale('log')
            plt.legend(fancybox=True, loc='upper right')
            if plot:
                plt.show()
            else:
                plt.savefig(self.save_path + "/precision_optimisation/lifetime_" + str(trials) + ".png", dpi=300)
            plt.clf()

            return tau_bound[0] / tau_bound[1]

        def calc_per_roi(area_size):
            x_percent_roi = [0.4, 0.4, 0.5, 0.6, 0.6]
            y_percent_roi = [0.4, 0.6, 0.5, 0.4, 0.6]

            min_locs = []
            tau_ub = []
            tau_b = []
            ratio = []
            ub_lt = np.array([])
            b_lt = np.array([])
            for i in range(len(x_percent_roi)):
                result, lifetime = qPAINT.td_tb_ratio_check(self, area_size, x_percent_roi[i], y_percent_roi[i], True,
                                                            area_size, i, plot_trace=True)

                min_locs.append(result[0])
                tau_ub.append(result[1])
                tau_b.append(result[2])
                ratio.append(result[3])

                ub_lt = np.append(ub_lt, lifetime[0])
                b_lt = np.append(b_lt, lifetime[1])

            np.save(self.save_path + "/precision_optimisation/ub_lifetimes.npy", ub_lt)
            np.save(self.save_path + "/precision_optimisation/b_lifetimes.npy", b_lt)
            plot_lifetime(ub_lt, b_lt, False, area_size)

            df = pd.DataFrame({'roi': np.arange(len(x_percent_roi))})
            df["min_locs"] = min_locs
            df["tau_ub"] = tau_ub
            df["tau_b"] = tau_b
            df["tau_ub/tau_b"] = ratio

            print("Area size " + str(area_size) + " nm: " + str(df["tau_ub/tau_b"].mean()))
            print(df)
            df.to_pickle(self.save_path + "/precision_optimisation/opt" + str(area_size) + ".pkl")

            return df["tau_ub/tau_b"].mean()

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        if not os.path.exists(self.save_path + "/precision_optimisation"):
            os.mkdir(self.save_path + "/precision_optimisation")

        if multiple_roi:
            complete = False
            while not complete:
                try:
                    calc_per_roi(initial_val)
                    np.save(self.save_path + "/precision_optimisation/optimized_area_for_analysis.npy", initial_val)
                    complete = True
                except AttributeError:
                    initial_val = initial_val - step_size
                    print(sys.exc_info()[0])
                    print("New initial value: " + str(initial_val) + " nm")
                except ValueError:
                    initial_val = initial_val + 10
                    print(sys.exc_info()[0])
                    print("New initial value: " + str(initial_val) + " nm")

    def compute_ub_lt(self, area_size, n_x, n_y, exclude_censored_lifetimes, roi_id, save_path, plot_trace=False,
                      compute_density=True):

        def select_pts(pts, frame, bound):
            x_min, x_max, y_min, y_max = bound
            cond_x = (pts[:, 0] <= x_max) & (pts[:, 0] >= x_min)
            cond_y = (pts[:, 1] <= y_max) & (pts[:, 1] >= y_min)
            pts_roi = pts[cond_x & cond_y]
            frame = frame[cond_x & cond_y]

            return pts_roi, frame

        def plot_grid(pts, grid, roi, path):
            _, _ = plt.subplots(figsize=(6, 6))
            plt.scatter(pts[:, 0], pts[:, 1], s=0.5)
            for n in range(grid[0].size):
                plt.axvline(grid[0][n], ls="--", c="k")
                plt.axhline(grid[1][n], ls="--", c="k")
            plt.xlim([grid[0][0], grid[0][-1]])
            plt.ylim([grid[1][0], grid[1][-1]])
            plt.xlabel("x (nm)")
            plt.ylabel("y (nm)")
            plt.savefig(path + "/grids" + str(roi) + ".png", dpi=300)
            plt.clf()

        def plot_traces(n_frame, single_signal, trial, path):
            frame_num = np.arange(0, n_frame, 1)
            plt.plot(frame_num, single_signal)
            plt.xlabel("Frame")
            plt.ylabel("Signal")
            plt.yticks(np.arange(2))
            plt.savefig(path + "/" + str(trial) + "_traces0.png", dpi=300)
            plt.clf()
            # plt.show()

        def compute_n_binders(area, tau_ub, conc, k_on=2 * 10 ** 6):
            n = 1 / (k_on * conc * tau_ub)
            total_area = (area * 10 ** -3) ** 2
            n_per_area = n / total_area  # um^-2

            return n_per_area

        results = []

        raw_pts = np.stack((self.x, self.y), axis=-1)
        bounds = find_roi(n_x, n_y, 5000)
        pts_in_roi, frames = select_pts(raw_pts, self.frame, bounds)

        x_grid = np.arange(bounds[0], bounds[1] + 1, area_size)
        y_grid = np.arange(bounds[2], bounds[3] + 1, area_size)
        plot_grid(pts_in_roi, [x_grid, y_grid], roi_id, save_path)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Compute traces from areas chosen
        signal = np.zeros(((x_grid.size - 1) * (y_grid.size - 1), self.n_frame), dtype=np.int64)
        ub_lt = np.array([])
        b_lt = np.array([])
        k = 0
        for i in range(x_grid.size - 1):
            for j in range(y_grid.size - 1):
                ratio = 0
                result = []
                bounding_box = [xx[i, j], xx[i, j + 1], yy[i, j], yy[i + 1, j]]
                pts_box, frame_box = select_pts(pts_in_roi, frames, bounding_box)

                if pts_box.size != 0:
                    signal[k, frame_box.astype(int)] = 1

                    _, count = np.unique(frame_box, return_counts=True)
                    if np.any(count != 1):
                        ratio = count[count != 1].size / count.size
                    result.append(ratio)

                    # Merge events that have less than 1 frames of dark time between bright time
                    signal = merge_event(frame_box.astype(int), None, signal, k, 2)
                    signal = merge_event(frame_box.astype(int), None, signal, k, 3)

                    dark_t, bright_t, _ = compute_lifetime(signal[k, :], exclude_censored_lifetimes)
                    n_events = dark_t.size + bright_t.size
                    result.append(n_events)

                    if dark_t.size > 3 or bright_t.size > 3:
                        lifetime = [dark_t, bright_t]
                        for m in range(2):
                            tau, f = ecdf(lifetime[m] / self.frame_rate)
                            param_bounds = ([0, 0], [1, np.inf])
                            pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
                            tau_bound1 = 1 / pred[0][1]
                            result.append(tau_bound1)
                            if m == 0 and compute_density:
                                density = compute_n_binders(area_size, tau_bound1, 250 * 10 ** -12)
                                result.append(density)

                        ub_lt = np.append(ub_lt, dark_t)
                        b_lt = np.append(b_lt, bright_t)
                        results.append(result)
                k += 1

        if plot_trace:
            plot_traces(self.n_frame, signal[0, :], area_size, save_path)

        return results, [ub_lt, b_lt]

    def run_qPAINT(self, area):

        def plot_lifetime(ub_lifetimes, b_lifetimes, plot, trials, path_save):
            lifetime = [ub_lifetimes, b_lifetimes]
            labels = ["Unbound state lifetimes", "Bound state lifetimes"]
            col = ["blue", "red"]
            tau_bound = []
            for j in range(2):
                tau, f = ecdf(lifetime[j] / self.frame_rate)
                plt.scatter(tau, 1 - f, lw=2, label=labels[j], color=col[j], s=1)

                param_bounds = ([0, 0], [1, np.inf])
                pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
                tau_bound.append(1 / pred[0][1])
                # print(tau_bound1)
                plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]), color=col[j], ls="--")
            plt.ylim([0.01, 1])
            plt.xlabel("State lifetimes (s)")
            plt.ylabel('1 - cdf')
            plt.yscale('log')
            plt.legend(fancybox=True, loc='upper right')
            if plot:
                plt.show()
            else:
                plt.savefig(path_save + "/lifetime_" + str(trials) + ".png", dpi=300)
            plt.clf()

        def calc_per_roi(area_size, save_path):
            x_percent_roi = [0.4, 0.4, 0.5, 0.6, 0.6]
            y_percent_roi = [0.4, 0.6, 0.5, 0.4, 0.6]

            double_event = []
            min_locs = []
            tau_ub = []
            tau_b = []
            density = []
            ub_lt = np.array([])
            b_lt = np.array([])
            for i in range(len(x_percent_roi)):
                result, lifetime = qPAINT.compute_ub_lt(self, area_size, x_percent_roi[i], y_percent_roi[i], True,
                                                        i, save_path)

                for k in range(len(result)):
                    data = result[k]
                    double_event.append(data[0])
                    min_locs.append(data[1])
                    tau_ub.append(data[2])
                    tau_b.append(data[3])
                    density.append(data[4])

                ub_lt = np.append(ub_lt, lifetime[0])
                b_lt = np.append(b_lt, lifetime[1])

            plot_lifetime(ub_lt, b_lt, False, area_size, save_path)

            df = pd.DataFrame({"mult_locs_ratio": double_event})
            df["n_events"] = min_locs
            df["tau_ub"] = tau_ub
            df["tau_b"] = tau_b
            df["density"] = density

            print("Average density: " + str(df["density"].mean()) + " um^-2")
            df.to_pickle(save_path + "/analysis.pkl")

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        path = self.save_path + "/qPAINT"
        if not os.path.exists(self.save_path + "/qPAINT"):
            os.mkdir(self.save_path + "/qPAINT")

        calc_per_roi(area, path)

    def run_calibration(self, area):

        def plot_lifetime(ub_lifetimes, b_lifetimes, plot, trials, path_save):
            lifetime = [ub_lifetimes, b_lifetimes]
            labels = ["Unbound state lifetimes", "Bound state lifetimes"]
            col = ["blue", "red"]
            tau_all = []
            for j in range(2):
                tau, f = ecdf(lifetime[j] / self.frame_rate)
                plt.scatter(tau, 1 - f, lw=2, label=labels[j], color=col[j], s=1)

                param_bounds = ([0, 0], [1, np.inf])
                pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
                tau_all.append(1 / pred[0][1])
                # print(tau_bound)
                plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]), color=col[j], ls="--")
            plt.ylim([0.01, 1])
            plt.xlabel("State lifetimes (s)")
            plt.ylabel('1 - cdf')
            plt.yscale('log')
            plt.legend(fancybox=True, loc='upper right')
            if plot:
                plt.show()
            else:
                plt.savefig(path_save + "/lifetime_" + str(trials) + ".png", dpi=300)
            plt.clf()
            return tau_all

        def calc_per_roi(area_size, save_path):
            x_percent_roi = [0.4, 0.4, 0.5, 0.6, 0.6]
            y_percent_roi = [0.4, 0.6, 0.5, 0.4, 0.6]

            double_event = []
            min_locs = []
            tau_ub = []
            tau_b = []
            ub_lt = np.array([])
            b_lt = np.array([])
            for i in range(len(x_percent_roi)):
                result, lifetime = qPAINT.compute_ub_lt(self, area_size, x_percent_roi[i], y_percent_roi[i], True,
                                                        i, save_path, compute_density=False)

                for k in range(len(result)):
                    data = result[k]
                    double_event.append(data[0])
                    min_locs.append(data[1])
                    tau_ub.append(data[2])
                    tau_b.append(data[3])

                ub_lt = np.append(ub_lt, lifetime[0])
                b_lt = np.append(b_lt, lifetime[1])

            tau_all = plot_lifetime(ub_lt, b_lt, False, area_size, save_path)

            df = pd.DataFrame({"mult_locs_ratio": double_event})
            df["n_events"] = min_locs
            df["tau_ub"] = tau_ub
            df["tau_b"] = tau_b

            print("Average tau_unbound: " + str(df["tau_ub"].mean()) + " s")
            print(tau_all[0])
            df.to_pickle(save_path + "/analysis.pkl")

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        path = self.save_path + "/calibration"
        if not os.path.exists(path):
            os.mkdir(path)

        if area is None:
            area = np.load(self.save_path + "/precision_optimisation/optimized_area_for_analysis.npy")
        print(area)
        calc_per_roi(area, path)

    def compile_calibration(self, conc, area):
        def linear_eq(x, a):
            output = a * x
            return output

        # for path in self.path_lists:
        #     df = pd.read_pickle(path + "/qPAINT/calibration/analysis.pkl")
        #     print(df["n_events"].describe())

        conc = np.array(conc)
        x_plot = conc / (10 ** -12)
        conc_all = np.array([])
        data_all = np.array([])
        mean = np.array([])
        std = np.array([])
        tau_b = np.array([])
        i = 0
        for path in self.path_lists:
            df = pd.read_pickle(path + "/qPAINT/calibration/analysis.pkl")
            tau_b = np.append(tau_b, df["tau_b"].mean())
            data = 1 / df["tau_ub"]
            data_all = np.append(data_all, 1 / df["tau_ub"].values)
            mean = np.append(mean, data.mean())
            print(data.mean(), 1 / df["tau_ub"].mean())
            std = np.append(std, data.std())
            conc_all = np.append(conc_all, np.repeat(conc[i], data.size))
            i += 1

        fig, ax = plt.subplots(ncols=2, sharey='all', sharex='all')
        ax[0].errorbar(x_plot, mean * (10 ** 2), yerr=std * (10 ** 2), capsize=5, fmt=".", ls=None, c="k")
        pred = scipy.optimize.curve_fit(linear_eq, conc_all, data_all, np.array([10 ** 6]))
        print(pred[0])
        ax[0].plot(np.arange(25, 400, 1), linear_eq(np.arange(25, 400, 1) * (10 ** -12), pred[0][0]) * (10 ** 2), "r--")
        ax[1].plot(x_plot, 1 / tau_b, "b.")
        ax[0].set_ylabel(r'1 / $\tau_{unbound}$' ' ($10^{-2} s^{-1}$)')
        ax[1].set_ylabel(r'$k_{off}$' ' ($s^{-1}$)')
        fig.suptitle(str(area))
        plt.xlim(left=0)
        plt.savefig(str(area) + "_calibration.png", dpi=300)
        # plt.show()


def density_filtering(labels, count_threshold=100):
    val, count = np.unique(labels, return_counts=True)

    cond = (count >= count_threshold)
    val = val[cond]
    id_filter = np.array([])
    for i in val:
        id_filter = np.append(id_filter, np.where(labels == i)[0])
    id_filter = np.sort(id_filter, axis=None).astype(int)
    labels = labels[id_filter]
    n_clusters_ = np.unique(labels).size
    print("number of estimated clusters after filtering : %d" % n_clusters_)

    return val


def plot_single_cluster(k, labels, cluster_centers, pts, col, radius):
    if ~isinstance(k, int):
        k = int(k)
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.scatter(pts[my_members, 0], pts[my_members, 1], s=1, color=col, marker=".")
    plt.axis("equal")
    result = np.zeros(4)
    result[0] = cluster_center[0] + 1.5 * radius
    result[1] = cluster_center[0] - 1.5 * radius
    result[2] = cluster_center[1] + 1.5 * radius
    result[3] = cluster_center[1] - 1.5 * radius

    return result


def load_data_from_folder(folder_path):
    path = folder_path + "/PreProcessing/FilteredData/"
    x = np.load(path + "x.npy")
    y = np.load(path + "y.npy")
    frame = np.load(path + "frame.npy")
    intensity = np.load(path + "intensity.npy")
    loc_precision = np.load(path + "loc_precision.npy")

    return x, y, frame, intensity, loc_precision


def merge_event(find_frames, find_intensity, signal, k, off_time_len):
    find_short_event = np.where((np.diff(find_frames) == off_time_len))[0]
    if find_short_event.size != 0:
        short_event = find_frames[find_short_event] + 1
        if find_intensity is not None:
            intensity_event = np.stack((find_intensity[find_short_event], find_intensity[find_short_event + 1]), axis=-1)
            intensity_event = np.mean(intensity_event, axis=1)
        else:
            intensity_event = 1

        try:
            signal[k, short_event] = intensity_event
        except IndexError:
            signal[short_event] = intensity_event
        if off_time_len == 3:
            try:
                signal[k, short_event + 1] = intensity_event
            except IndexError:
                signal[short_event + 1] = intensity_event

    return signal


def compute_lifetime(state_arr_original, exclude_censored_lifetimes):
    if np.any(state_arr_original > 1):
        state_arr = np.copy(state_arr_original)
        state_arr[state_arr != 0] = 1
    else:
        state_arr = state_arr_original

    bound_lifetimes = np.array([])
    unbound_lifetimes = np.array([])
    intensity_summary = {}
    i = 0
    k = 0
    while i < state_arr.shape[0] - 1:
        cur_state = state_arr[i]
        intensity = []
        lifetime = 0
        while cur_state == state_arr[i] and i < state_arr.shape[0] - 1:
            intensity.append(state_arr_original[i])
            lifetime = lifetime + 1
            i = i + 1
        if cur_state == 0:
            unbound_lifetimes = np.append(unbound_lifetimes, lifetime)
        elif cur_state == 1:
            bound_lifetimes = np.append(bound_lifetimes, lifetime)
            intensity_summary[k] = intensity
            k += 1

    if exclude_censored_lifetimes:
        if state_arr[0] == 0:
            unbound_lifetimes = unbound_lifetimes[1:]
        elif state_arr[0] == 1:
            bound_lifetimes = bound_lifetimes[1:]
            intensity_summary.pop(0)
            i = 0
            for k in np.arange(1, len(intensity_summary) + 1):
                intensity_summary[i] = intensity_summary.pop(k)
                i += 1

        if state_arr[state_arr.shape[0] - 1] == 0:
            unbound_lifetimes = unbound_lifetimes[:-1]
        elif state_arr[state_arr.shape[0] - 1] == 1:
            if bound_lifetimes.size != 0:
                intensity_summary.pop(bound_lifetimes.size - 1)
            bound_lifetimes = bound_lifetimes[:-1]

    return unbound_lifetimes, bound_lifetimes, intensity_summary


def ecdf(sample_):
    # convert sample to a numpy array, if it isn't already
    sample_ = np.atleast_1d(sample_)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample_, return_counts=True)
    if quantiles:
        quantiles = np.insert(quantiles, 0, quantiles[0] - 0.1)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample_.size
    cumprob = np.insert(cumprob, 0, 0)

    return quantiles, cumprob


def exp_lifetimes(x, a, b):
    output = a * np.exp(-b * x)
    return output


def exp_lifetimes2(x, a, b, c, d):
    output = a * np.exp(-b * x) + ((1 - a - c) * np.exp(-d * x))
    return output


def exp_lifetimes3(x, a, b, c, d, e, f):
    output = a * np.exp(-b * x) + ((1 - a - c) * np.exp(-d * x)) + ((1 - a - c - e) * np.exp(-f * x))
    return output


def exp_lifetimes2_noc(x, a, b, d):
    output = a * np.exp(-b * x) + ((1 - a) * np.exp(-d * x))
    return output


def find_clust_with_n_bright_time(bright_time, n):
    n = int(n)
    if (sum(bright_time == n) != 0) and (sum(bright_time == n) == bright_time.size):
        return True


def find_roi(n_x, n_y, roi_size=5000):
    fov_x, fov_y = 50000, 60000
    x_min, x_max = fov_x * n_x - roi_size / 2, fov_x * n_x + roi_size / 2
    y_min, y_max = fov_y * n_y - roi_size / 2, fov_y * n_y + roi_size / 2
    return x_min, x_max, y_min, y_max


def clark_evans_test(pts, choose_roi=True, n_x=0.5, n_y=0.5, N=1000, sample_percent=0.3, roi_size=5000):
    from pointpats import PointPattern

    try:
        nnd = PointPattern(pts).nnd

        # Exclude points outside bounding box
        if choose_roi:
            x_min, x_max, y_min, y_max = find_roi(n_x, n_y, roi_size=roi_size)
            square_size = (roi_size + 500) ** 2
        else:
            bounding_box = 0.1
            x_min, x_max, y_min, y_max = bounding_box, 50000 * (1 - bounding_box), bounding_box, 60000 * (
                        1 - bounding_box)
            square_size = 50000 * 60000
        cond_x = (pts[:, 0] <= x_max) & (pts[:, 0] >= x_min)
        cond_y = (pts[:, 1] <= y_max) & (pts[:, 1] >= y_min)
        nnd = nnd[cond_x & cond_y]

        lambda_window = pts.shape[0] / square_size
        mu = 1 / (2 * np.sqrt(lambda_window))

        if len(nnd) > 3:
            m = int(sample_percent * len(nnd))
        else:
            m = len(nnd)
        std = np.sqrt((4 - np.pi) / (m * 4 * np.pi * lambda_window))
        z_score_increment = np.zeros(N)
        for i in range(N):
            id_random = np.random.randint(len(nnd), size=m)
            mean_d = nnd[id_random].mean()
            z_score_increment[i] = (mean_d - mu) / std
        # plt.hist(z_score_increment, bins=100)
        # plt.show()

        return z_score_increment.mean()
    except ZeroDivisionError:
        pass


def qPAINT_calibration_single(paths, framerate, imager_conc):
    for path in paths:
        def merge_event_calibration(signal_clust, frame_index, off_time_len):
            find_short_event = np.where((np.diff(frame_index) == off_time_len))[0]
            if find_short_event.size != 0:
                short_event = frame_index[find_short_event] + 1
                signal_clust[0, short_event] = 1
                if off_time_len == 3:
                    signal_clust[0, short_event + 1] = 1
            return signal_clust

        n_frame = np.load(path + "/PreProcessing/FilteredData/n_frame.npy")
        signal = np.zeros((1, n_frame), dtype=int)
        for i in range(5):
            ms_labels = np.load(path + "/Clustering/roi" + str(i) + "/ms_labels.npy")
            frame = np.load(path + "/Clustering/roi" + str(i) + "/ms_frames.npy").astype(int)
            labels_roi, count = np.unique(ms_labels, return_counts=True)
            label = labels_roi[(count > 15) & (count < 30)]
            for j in label:
                signal_j = np.zeros((1, n_frame))
                find_frames = frame[ms_labels == j]
                signal_j[0, find_frames] = 1
                signal_j = merge_event_calibration(signal_j, find_frames, 2)
                signal_j = merge_event_calibration(signal_j, find_frames, 3)
                signal = np.append(signal, signal_j, axis=0)
        signal = np.delete(signal, 0, 0)

        unbound_LT = np.array([])
        for i in range(signal.shape[0]):
            dark_t, bright_t, _ = compute_lifetime(signal[i, :], True)
            unbound_LT = np.append(unbound_LT, dark_t)

        tau, f = ecdf(unbound_LT / framerate)
        plt.scatter(tau, 1 - f, lw=2, label='Empirical CDF', color='red', s=1)

        param_bounds = ([0, 0], [1, np.inf])
        pred = scipy.optimize.curve_fit(exp_lifetimes, tau, 1 - f, np.array([0.1, 0.1]), bounds=param_bounds)
        tau_bound1 = 1 / pred[0][1]
        print(tau_bound1, pred[0][1] / imager_conc / 10 ** 6)
        plt.plot(tau, exp_lifetimes(tau, pred[0][0], pred[0][1]))

        plt.ylim([0.01, 1])
        plt.xlabel("Unbound lifetime (s)")
        plt.ylabel('1 - cdf')
        plt.yscale('log')
        plt.legend(fancybox=True, loc='upper right')
        plt.title("Calibration qPAINT")
        plt.savefig("Calibration_qPAINT.png", dpi=300)
