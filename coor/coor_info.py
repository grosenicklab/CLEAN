import os
import pandas as pd


# For each treatment session, calculate the mean, median, standard deviation,
# min, max, and range of coil position coordinates x, y, and z
# then write these values to a new file (one file per patient)
def write_stats(samples_df, patient_id, data_folder):

    # Iterate over dates in samples_df
    for day in samples_df['Date'].unique():
        if day is not None:
            day_sessions = samples_df[samples_df['Date'] == day]

            # Calculate mean, median, and standard deviation
            mean_xyz = day_sessions[['Loc. X', 'Loc. Y', 'Loc. Z']].mean()
            median_xyz = day_sessions[['Loc. X', 'Loc. Y', 'Loc. Z']].median()
            std_xyz = day_sessions[['Loc. X', 'Loc. Y', 'Loc. Z']].std()

            # Calculate min, max, and range
            min_xyz = day_sessions[['Loc. X', 'Loc. Y', 'Loc. Z']].min()
            max_xyz = day_sessions[['Loc. X', 'Loc. Y', 'Loc. Z']].max()
            range_xyz = max_xyz - min_xyz

            # Save sample statistics as a dictionary
            day_stats = {
                'mean': mean_xyz,
                'median': median_xyz,
                'sd': std_xyz,
                'min': min_xyz,
                'max': max_xyz,
                'range': range_xyz,
                'patient': patient_id,
                'date': day
            }

            # Convert dict to dataframe for readability
            day_stats_df = pd.DataFrame(day_stats)

            # Get current path and set to output_files directory
            parent_directory = os.path.dirname(data_folder)
            save_path = os.path.join(parent_directory, 'output_files', 'stats')

            # If the file does not exist, write a new file
            # Otherwise, append to patient stats file for every iteration according to day
            with open(f'{save_path}/coor_stats_{patient_id}.txt', mode='a+') as statsfile:
                statsfile.write(f'{day}\n')
                statsfile.write(f'{str(day_stats_df)}\n\n')

    print(f'Writing stats file for patient {patient_id} to: {save_path}')  # noqa: T201
