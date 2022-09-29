# DNA-PAINT

Python scripts to analyse DNA-PAINT localization data

PAINT_analysis_functions contains classes and functions that analyse raw ThunderSTORM localization data. It contains the following modules:
1. DriftCorrection: Correct mechanical drift using localizations of fiducial markers
2. Pre-filtering/pre-processing: Filters localizations based on criteria on uncertainty (localization precision), offset (signal-noise ratio) and sigma (PSF width)
3. MergedLoc: Merge 5-min blocks of localizations to analyse as a whole
4. Clustering: Perform mean shift clustering to localizations to determine approximate locations of binder
5. LifetimeAnalysis: Merge localizations within cluster into events and analyse the bright and dark lifetimes
6. MoleculeDensity: Determine density of binders
7. ClarkEvansTest: Determine distribution of binders

PAINT_block contains code to analyse 5-min block of localization data

PAINT_merged contains code to analyse merged localization data

PAINT_plot_functions contains classes and functions to extract analysed parameters from results folder and plotting these parameters

PAINT_plot contains code to plot the following parameters: "tau_bright", "tau_dark", "density", "z_scores", "density_filter_repeatedframe", "z_scores_filter_repeatedframe"

test contains short snippets of code for testing new functions
