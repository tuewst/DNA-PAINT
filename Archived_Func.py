""" Archived class functions

Last updated: 19 Dec 2022
This script contains archived class functions that are no longer in use in the analysis of DNA-PAINT experiments.

The script contains the following functions:

    * Autocorrelation class: The class implements lbFCS to analyse DNA-PAINT data and to perform molecular counting
"""


# noinspection PyTypeChecker
class Autocorrelation:
    def __init__(self, path):
        path_signal = path + "/LifetimeAnalysis"
        self.traces = np.load(path_signal + "/signal.npy")
        self.path = path + "/Autocorrelation"

    def run_per_clust(self):
        import lbFCS

        trace = self.traces[388, :]

        ac = lbFCS.autocorrelate(trace, m=16, deltat=1, normalize=True, copy=False, dtype=np.float64())
        mono_A, mono_tau, mono_chi = lbFCS.fit_ac_mono(ac)  # Get fit
        mono_A_lin, mono_tau_lin = lbFCS.fit_ac_mono_lin(ac)  # Get fit

        Autocorrelation.plot_autocorrelation(self, clust_num, ac[1:-15, 0], ac[1:-15, 1], mono_A, mono_tau, mono_A_lin,
                                             mono_tau_lin, True)

    def plot_autocorrelation(self, k, t, g, mono_A, mono_tau, mono_A_lin, mono_tau_lin, save):
        import lbFCS

        plt.plot(t, lbFCS.ac_mono_exp(t, mono_A, mono_tau), '-', lw=2, c='r')
        plt.plot(t, lbFCS.ac_mono_exp(t, mono_A_lin, mono_tau_lin), '-', lw=2, c='b')
        plt.plot(t, g, ".")
        plt.axhline(1, ls='--', lw=2, color='k')
        plt.xscale('symlog')
        plt.xlim(left=np.min(t) - 0.5)
        # plt.xticks([])
        # plt.yticks([])
        plt.xlabel("$t$")
        plt.ylabel("$G_{i} (t)$")
        if save:
            plt.savefig(self.path + "/autocorrelation_" + str(k) + ".png", dpi=300)
            plt.clf()
            plt.close()
        else:
            plt.show()

    def run(self):
        import lbFCS

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        n_clust = self.traces.shape[0]
        mono_A = np.zeros(n_clust)
        mono_tau = np.zeros(n_clust)
        mono_chi = np.zeros(n_clust)
        mono_A_lin = np.zeros(n_clust)
        mono_tau_lin = np.zeros(n_clust)

        # n_plot = np.random.choice(np.arange(n_clust), size=15)

        for clust_num in tqdm(range(n_clust)):
            trace = self.traces[clust_num, :]

            try:
                ac = lbFCS.autocorrelate(trace, m=16, deltat=1, normalize=True, copy=False, dtype=np.float64())
                mono_A[clust_num], mono_tau[clust_num], mono_chi[clust_num] = lbFCS.fit_ac_mono(ac)  # Get fit
                # mono_A_lin[clust_num], mono_tau_lin[clust_num] = lbFCS.fit_ac_mono_lin(ac)  # Get fit

                # mono_A, mono_tau, mono_chi = lbFCS.fit_ac_mono(ac)  # Get fit
                # mono_A_lin, mono_tau_lin = lbFCS.fit_ac_mono_lin(ac)  # Get fit

                # t[clust_num, :] = ac[1:-15, 0]
                # g[clust_num, :] = ac[1:-15, 1]

                # if clust_num in n_plot:
                Autocorrelation.plot_autocorrelation(self, clust_num, ac[1:-15, 0], ac[1:-15, 1], mono_A[clust_num],
                                                     mono_tau[clust_num], mono_A_lin[clust_num],
                                                     mono_tau_lin[clust_num], True)
                # Autocorrelation.plot_autocorrelation(self, clust_num, ac[1:-15, 0], ac[1:-15, 1], mono_A,
                #                                      mono_tau, mono_A_lin,
                #                                      mono_tau_lin, True)
            except AssertionError:
                print("Cluster %d cannot be normalized" % clust_num)

        np.save(self.path + "/mono_A.npy", mono_A)
        np.save(self.path + "/mono_tau.npy", mono_tau)
        # np.save(self.path + "/mono_chi.npy", mono_chi)
        # np.save(self.path + "/mono_A_lin.npy", mono_A_lin)
        # np.save(self.path + "/mono_tau_lin.npy", mono_tau_lin)
