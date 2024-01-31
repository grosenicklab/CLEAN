import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Plots the coil x, y, and z positions each day,
# averaging over the 10 sessions in the day
def plot_coor_days(samples_df, patient_id, data_folder):

    df_melt = pd.melt(samples_df, id_vars=['Index', 'Date'],
                      value_vars=['Loc. X', 'Loc. Y', 'Loc. Z'],
                      var_name='Coordinate', value_name='Value')

    df_melt_grid = sns.FacetGrid(df_melt, col='Date', hue='Coordinate')
    df_melt_grid.map(sns.lineplot, 'Index', 'Value')

    # Set plot layout, axes labels, titles, and add legend
    plt.tight_layout()
    df_melt_grid.set_axis_labels('Sample', 'Position (mm)')
    df_melt_grid.set_titles('{col_name}')
    df_melt_grid.fig.subplots_adjust(top=0.8)
    df_melt_grid.fig.suptitle(f'Coil Position Changes for {patient_id}')
    df_melt_grid.add_legend()

    # Save figure
    parent_directory = os.path.dirname(data_folder)
    save_path = os.path.join(parent_directory, 'output_files/plots/days')
    print(f'Saving coor days plot for patient {patient_id} to: {save_path}')
    plt.savefig(f'{save_path}/coor_days_{patient_id}.png')


# Plots the coil x, y, and z positions over the course of each treatment
# for all treatment sessions in the available date range:
# 10 sessions/day, 5 days total
def plot_coor_sessions(samples_df, patient_id, data_folder):

    # Iterate over dates in samples_df
    for day in samples_df['Date'].unique():
        if day is not None:
            day_sessions = samples_df[samples_df['Date'] == day]

            # Melt dataframe to correct format for use with seaborn
            day_melted_df = pd.melt(day_sessions, id_vars=['Index', 'Session Name'],
                                    value_vars=['Loc. X', 'Loc. Y', 'Loc. Z'],
                                    var_name='Coordinate', value_name='Value')

            # Create grid for plotting multiple panels on one figure
            # Map lineplot onto each panel
            day_grid = sns.FacetGrid(day_melted_df, col='Session Name',
                                     col_wrap=5, hue='Coordinate')
            day_grid.map(sns.lineplot, 'Index', 'Value')

            # Set plot layout, axes labels, titles, and add legend
            plt.tight_layout()
            day_grid.set_axis_labels('Sample', 'Position (mm)')
            day_grid.set_titles('{col_name}')
            day_grid.fig.subplots_adjust(top=0.9)
            day_grid.fig.suptitle(f'Coil Position Changes per Session for {patient_id}, {day}')
            day_grid.add_legend()

            # Save figure
            parent_directory = os.path.dirname(data_folder)
            save_path = os.path.join(parent_directory, 'output_files/plots/sessions')
            print(f'Saving coor sessions plot for patient {patient_id}, date {day} to: {save_path}')
            plt.savefig(f'{save_path}/coor_session_{patient_id}_{day}.png')
