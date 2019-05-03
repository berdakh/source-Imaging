# EEG processing and Movement related cortical potentials! 
import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt 
#import glob, os
   
#%% path to the EEG data 
subjects_dir = 'C:\\Users\\Berdakh\\Google Drive\\uhdata\\data\\Subject_S9009'
filename = 'S9011_ses1_cond1_block0001.vhdr'

#os.chdir(subjects_dir)
#all_files = glob.glob("*.vmrk") # find all files with .vmrk extension
fname = op.join(subjects_dir, filename)
d = mne.io.read_raw_brainvision(fname, preload=True)

#%% Removing power-line noise with notch filtering
picks = mne.pick_types(d.info, eeg=True, eog=False,stim=False, exclude='bads')
d.notch_filter(np.arange(60, 241, 60), picks=picks, filter_length='auto',phase='zero')
# plot events 
##plt.plot(d._data[-1])
#%% Get standard electrode positions 
montage = mne.channels.read_montage(kind = 'standard_1020', ch_names = d.info['ch_names'])
d.set_montage(montage)
# One could also load subject specific 3D electrodes using the mne.channels.montage.read_dig_montage
# elec = mne.channels.montage.read_dig_montage(hsp=None, hpi=None, elp='S9011_ses14_electrodefile.elp',point_names=ch)

#%% 
# High-pass filtering to remove slow drifts
# To remove slow drifts, you can high pass.
d.filter(0.1, None, l_trans_bandwidth='auto', filter_length='auto',
           phase='zero')
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
#%%##############################################################################
# Removing power-line noise with low-pass filtering
# low pass filtering below 50 Hz
d.filter(None, 40., h_trans_bandwidth='auto', filter_length='auto',
           phase='zero')
#d.plot_psd(area_mode='range', tmax=10.0, picks=picks)
#%%##############################################################################
# Downsampling and decimation
#d.resample(60, npad="auto")  # set sampling frequency to 100Hz
ch_names = ['TP7','TP9', 'FT9', 'FT10', 'TP10']

#drop_ch_names =  ['L_B1', 'L_B2', 'L_T1', 'L_T2', 'R_B1', 'R_B2', 'R_T1', 'R_T2','STI 014']
d.drop_channels(ch_names)
d.plot_sensors(show_names = True)

#%% Artifact Correction with ICA
"""
ICA finds directions in the feature space
corresponding to projections with high non-Gaussianity. We thus obtain
a decomposition into independent components, and the artifact's contribution
is localized in only a small number of components.
These components have to be correctly identified and removed.

If EOG or ECG recordings are available, they can be used in ICA to
automatically select the corresponding artifact components from the
decomposition. To do so, you have to first build an Epoch object around
blink or heartbeat event.
"""
from mne.preprocessing import ICA
# ICA parameters:
n_components = 10  # if float, select n_components by explained variance of PCA
method = 'infomax'  # for comparison with EEGLAB try "extended-infomax" here
decim = 2.3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23
picks_eeg = mne.pick_types(d.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')
###############################################################################
# Define the ICA object instance
ica = ICA(n_components=n_components, random_state=random_state, method='infomax')
print(ica)

#%%##############################################################################
# we avoid fitting ICA on crazy environmental artifacts that would
# dominate the variance and decomposition
ica.fit(d, picks=picks_eeg, decim=3)
print(ica)
ica.plot_components()  # can you spot some potential bad guys?
#%%##############################################################################
# Component properties
# Let's take a closer look at properties of first three independent components.
# first, component 0:
#ica.plot_properties(d, picks=0)
# we can see that the data were filtered so the spectrum plot is not
# very informative, let's change that:
#ica.plot_properties(d, picks=0, psd_args={'fmax': 35.})
#%%##############################################################################
# we can also take a look at multiple different components at once:
#ica.plot_properties(d, picks=[1, 2], psd_args={'fmax': 35.})
#%% INTERACTIVE PLOT 
#ica.plot_components(picks=range(10), inst=d)

#%% Advanced artifact detection
#%% We simplify things by setting the maximum number of components to reject
eog_inds, scores = ica.find_bads_eog(d, ch_name = 'Fp1', threshold=1)  # find via correlation
#ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).
#ica.plot_sources(d, exclude=eog_inds)  # look at source time course
# We can take a look at the properties of that component, now using the
# data epoched with respect to EOG events. We will also use a little bit of smoothing along the trials axis in the
# epochs image:
#ica.plot_properties(d, picks=eog_inds, psd_args={'fmax': 35.},image_args={'sigma': 1.})
ica.exclude.extend(eog_inds)
# from now on the ICA will reject this component even if no exclude
# parameter is passed, and this information will be stored to disk
# on saving uncomment this for reading and writing
# ica.save('my-ica.fif')
# ica = read_ica('my-ica.fif')
d = ica.apply(d, exclude=eog_inds)
# Perform CAR operation
d.set_eeg_reference(ref_channels = "average", projection=False) 

#%% BAD CHANNEL IDENTIFICATION 
picks = mne.pick_types(d.info, meg=True, eeg=True, eog=True,
                       stim=False, exclude='bads')
baseline = (None, 0)  # means from the first instant to t = 0
# Define Epochs and compute an ERP for the movement onset condition.
events = mne.find_events(d, stim_channel=None, output='onset', 
                         consecutive='increasing', min_duration=0, 
                         shortest_event=1, mask=None,
                         uint_cast=False, mask_type='and', 
                         initial_event=False, verbose=None)

event_id, tmin, tmax = {'MRCP': 16}, -1, 1
reject = dict(eeg=50)
epochs_params = dict(events=events, event_id=16, 
                     tmin=tmin, tmax=tmax,reject=reject)

# Segment continuous EEG data around the event onset 
ep = mne.Epochs(d, **epochs_params,picks=picks, 
                baseline=baseline,preload=True) 
ep.drop_bad()

#%% 
# To do the low-pass and high-pass filtering in one step you can do
# a *band-pass* filter by running the following: band-pass filtering in the range .1 Hz - 4 Hz
ep.filter(0.1, 4., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
          filter_length='auto', phase='zero')

# interpolate bad channels if exist 
ep.interpolate_bads(reset_bads='True', mode = 'accurate')

#%% save the data as -epochs 
# mne.io.write_info('tmp-ave.fif', d)
#filename2save = "".join([filename.split(".")[0],'-epo.fif'])
#ep.save(filename2save)
