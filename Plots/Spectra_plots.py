import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
sns.set_context("paper", font_scale=1.5)

# import imgs_dict.npy for plotting:
dict_plot = np.load('spectra_dict.npy', allow_pickle=True).item()
del dict_plot['note']
del dict_plot['wavenumbers']

dict = np.load('spectra_dict.npy', allow_pickle=True).item()

wavenumbers  = dict['wavenumbers']
net_input = dict['Low SNR']

model_palette = {'Cycle': 'tab:blue', 
                 'SG': 'tab:orange', 
                 'AE': 'tab:green', 
                 'Wiener': 'tab:red', 
                 'Wavelet': 'tab:purple', 
                 'Low SNR': 'tab:gray', 
                 'High SNR': 'k'}

df = pd.DataFrame()
df['wavenumbers'] = wavenumbers

# use dict_plot keys and values to make new columns in spec_df
for key, value in dict_plot.items():
    df[key] = value[0]

# Assuming your dataframe is called 'df'
df_melted = pd.melt(df, id_vars='wavenumbers', var_name='Method', value_name='Intensity')

# Filter the dataframe to include only the 'High SNR' data
df_high_snr = df_melted[df_melted['Method'] == 'High SNR'].copy()

# Remove the 'High SNR' data from the main dataframe
df_melted = df_melted[df_melted['Method'] != 'High SNR'].copy()

# Create the FacetGrid and plot the spectra
g = sns.FacetGrid(df_melted, col='Method', col_wrap=3, hue='Method', palette=model_palette, height=4, aspect=1.5)

# Iterate over each method and plot the data
for method, ax in zip(df_melted['Method'].unique(), g.axes):
    # Plot the method data
    method_data = df_melted[df_melted['Method'] == method]
    plot_method = ax.plot(method_data['wavenumbers'], method_data['Intensity'], label=method, linewidth=2., color=model_palette[method])

    if method == 'Cycle':
        ax.text(0.05, 0.975, 'cycleGAN', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left', color=model_palette[method])
    else:
        # add the method name to the plot
        ax.text(0.05, 0.975, method, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left', color=model_palette[method])

    # Plot the 'High SNR' data on the same axes
    plot_high_snr = ax.plot(df_high_snr['wavenumbers'], df_high_snr['Intensity'], label='High SNR', linewidth=1.5, color='k', linestyle='--')

# Set labels and titles
g.set_axis_labels("Wavenumber [cm$^{-1}$]", "Intensity [a.u.]")
# g.set_titles("{col_name}")

# Adjust the plot layout
plt.tight_layout()

# put letter A in the top left corner outside the plot
g.fig.text(0.0, 1.05, 'A', fontsize=28, fontweight='bold', va='top', ha='left')

plt.savefig('spectra_A_.pdf', bbox_inches='tight')

#####################################################################################

df = pd.DataFrame()
df['wavenumbers'] = wavenumbers

# use dict_plot keys and values to make new columns in spec_df
for key, value in dict_plot.items():
    df[key] = value[1]

# Assuming your dataframe is called 'df'
df_melted = pd.melt(df, id_vars='wavenumbers', var_name='Method', value_name='Intensity')

# Filter the dataframe to include only the 'High SNR' data
df_high_snr = df_melted[df_melted['Method'] == 'High SNR'].copy()

# Remove the 'High SNR' data from the main dataframe
df_melted = df_melted[df_melted['Method'] != 'High SNR'].copy()

# Create the FacetGrid and plot the spectra
g = sns.FacetGrid(df_melted, col='Method', col_wrap=3, hue='Method', palette=model_palette, height=4, aspect=1.5)

# Iterate over each method and plot the data
for method, ax in zip(df_melted['Method'].unique(), g.axes):
    # Plot the method data
    method_data = df_melted[df_melted['Method'] == method]
    plot_method = ax.plot(method_data['wavenumbers'], method_data['Intensity'], label=method, linewidth=2., color=model_palette[method])

    if method == 'Cycle':
        ax.text(0.05, 0.975, 'cycleGAN', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left', color=model_palette[method])
    else:
        # add the method name to the plot
        ax.text(0.05, 0.975, method, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left', color=model_palette[method])

    # Plot the 'High SNR' data on the same axes
    plot_high_snr = ax.plot(df_high_snr['wavenumbers'], df_high_snr['Intensity'], label='High SNR', linewidth=1.5, color='k', linestyle='--')

# Set labels and titles
g.set_axis_labels("Wavenumber [cm$^{-1}$]", "Intensity [a.u.]")
# g.set_titles("{col_name}")

# Adjust the plot layout
plt.tight_layout()
# put letter A in the top left corner outside the plot
g.fig.text(0.0, 1.05, 'B', fontsize=28, fontweight='bold', va='top', ha='left')

plt.savefig('spectra_B_.pdf', bbox_inches='tight')


#####################################################################################

df = pd.DataFrame()
df['wavenumbers'] = wavenumbers

# use dict_plot keys and values to make new columns in spec_df
for key, value in dict_plot.items():
    df[key] = value[2]

# Assuming your dataframe is called 'df'
df_melted = pd.melt(df, id_vars='wavenumbers', var_name='Method', value_name='Intensity')

# Filter the dataframe to include only the 'High SNR' data
df_high_snr = df_melted[df_melted['Method'] == 'High SNR'].copy()

# Remove the 'High SNR' data from the main dataframe
df_melted = df_melted[df_melted['Method'] != 'High SNR'].copy()

# Create the FacetGrid and plot the spectra
g = sns.FacetGrid(df_melted, col='Method', col_wrap=3, hue='Method', palette=model_palette, height=4, aspect=1.5)

# Iterate over each method and plot the data
for method, ax in zip(df_melted['Method'].unique(), g.axes):
    # Plot the method data
    method_data = df_melted[df_melted['Method'] == method]
    plot_method = ax.plot(method_data['wavenumbers'], method_data['Intensity'], label=method, linewidth=2., color=model_palette[method])

    if method == 'Cycle':
        ax.text(0.05, 0.975, 'cycleGAN', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left', color=model_palette[method])
    else:
        # add the method name to the plot
        ax.text(0.05, 0.975, method, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left', color=model_palette[method])

    # Plot the 'High SNR' data on the same axes
    plot_high_snr = ax.plot(df_high_snr['wavenumbers'], df_high_snr['Intensity'], label='High SNR', linewidth=1.5, color='k', linestyle='--')

# Set labels and titles
g.set_axis_labels("Wavenumber [cm$^{-1}$]", "Intensity [a.u.]")
# g.set_titles("{col_name}")

# Adjust the plot layout
plt.tight_layout()
# put letter A in the top left corner outside the plot
g.fig.text(0.0, 1.05, 'C', fontsize=28, fontweight='bold', va='top', ha='left')

plt.savefig('spectra_C_.pdf', bbox_inches='tight')

#####################################################################################
#                            CYTO & NUC SPECTRA PLOT
#####################################################################################
cyto_dict = np.load('cyto_nuc_dict.npy', allow_pickle=True).item()
wavenumbers = cyto_dict['wavenumbers']
del cyto_dict['wavenumbers']

df = pd.DataFrame()
df['wavenumbers'] = wavenumbers
# use dict_plot keys and values to make new columns in spec_df
for key, value in cyto_dict.items():
    df[key] = value


sns.set_context('paper', font_scale=1.5)
# now do them as a subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ax[0].plot(df['wavenumbers'], df['High SNR nuc'], label='Ground truth', color=model_palette['High SNR'], linestyle='--')
ax[0].plot(df['wavenumbers'], df['Cycle nuc'], label='cycleGAN', color=model_palette['Cycle'])
ax[0].plot(df['wavenumbers'], df['Wavelet nuc'], label='Wavelet', color=model_palette['Wavelet'])
ax[0].set_xlabel('Wavenumber [cm$^{-1}$]')
ax[0].set_ylabel('Intensity [a.u.]')
ax[0].legend(frameon=False)
# ax[0].set_title('Nucleic Acid Spectra', loc='left')
ax[0].text(-0.15, 1.1, 'B', transform=ax[0].transAxes, fontsize=18, fontweight='bold', va='top')
ax[0].set_title('Nucleic acid', loc='left', fontweight='bold', fontsize=16)


ax[1].plot(df['wavenumbers'], df['High SNR cyt'], label='Ground truth', color=model_palette['High SNR'], linestyle='--')
ax[1].plot(df['wavenumbers'], df['Cycle cyt'], label='cycleGAN', color=model_palette['Cycle'])
ax[1].plot(df['wavenumbers'], df['Wavelet cyt'], label='Wavelet', color=model_palette['Wavelet'])
# ax[1].legend(frameon=False)
ax[1].set_xlabel('Wavenumber [cm$^{-1}$]')
# ax[1].set_ylabel('Intensity [a.u.]')
# ax[1].set_title('Cytoplasm Spectra', loc='left')
ax[1].text(-0.1, 1.1, 'C', transform=ax[1].transAxes, fontsize=18, fontweight='bold', va='top')
ax[1].set_title('Cytoplasm', loc='left', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.savefig('nuc_cyt_spectra_labeled_.pdf', bbox_inches='tight')

#####################################################################################
#                               BAR PLOTS
#####################################################################################

dict = np.load('MCSE_dict.npy', allow_pickle=True).item()

# if there's an 'std' within the dict keys, return the value:
models = [i for i in dict.keys() if not 'std' in i]
# strip '_MCSE' from the model names:
models = [i.replace('_MCSE', '') for i in models]

std = [dict[i] for i in dict.keys() if 'std' in i]
MCSE = [dict[i] for i in dict.keys() if not 'std' in i]

# create a dataframe:
df = pd.DataFrame({'models': models, 'MCSE': MCSE, 'MCSE_std': std})



dict2 = np.load('TMMSE_dict.npy', allow_pickle=True).item()
# if there's an 'std' within the dict keys, return the value:
models = [i for i in dict2.keys() if not 'std' in i]
# strip '_MCSE' from the model names:
models = [i.replace('_TMMSE', '') for i in models]

std = [dict2[i] for i in dict2.keys() if 'std' in i]
TMMSE = [dict2[i] for i in dict2.keys() if not 'std' in i]

# create a dataframe:
df2 = pd.DataFrame({'models': models, 'TMMSE': TMMSE, 'TMMSE_std': std})

########
# Plot
########

sns.set(style="darkgrid")
sns.set_context("paper", font_scale=1.5)
# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 5))

# Set the bar width and position offset for the grouped barplot
bar_width = 0.35
bar_positions = range(len(df['models']))

# Define a custom color palette for the models
model_palette = {'cycleGAN': 'tab:blue', 'SG': 'tab:orange', 'AE': 'tab:green', 'Wiener': 'tab:red', 'Wavelet': 'tab:purple'}

# Plot the first dataframe as grouped bars
ax.bar(bar_positions, df['MCSE'], yerr=df['MCSE_std'], width=bar_width,
       hatch='//', color=[model_palette[model] for model in df['models']],  label='MCSE', linewidth=1, alpha=0.8,
       capsize=4)

# Plot the second dataframe as grouped bars, slightly offset on the x-axis
ax.bar([pos + bar_width for pos in bar_positions], df2['TMMSE'], yerr=df2['TMMSE_std'], width=bar_width,
       hatch="\\\\", color=[model_palette[model] for model in df2['models']],  label='TMMSE', linewidth=1, alpha=0.8,
       capsize=4)

# Set the x-ticks and labels
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
ax.set_xticklabels(df['models'])
# ax.set_xlabel('Models')

# Set the y-axis label
ax.set_ylabel('Error [a.u.]')

legend = ax.legend(prop={'size': 15}, loc='upper left')
legend.get_frame().set_linewidth(0)  # Remove the legend border

# Customizing the legend patches to show hatches only (set the facecolor to 'none')
for patch in legend.get_patches():
    patch.set_facecolor('gray')

plt.tight_layout()

# Save the figure
plt.savefig('MCSE_TMMSE_.pdf', bbox_inches='tight')