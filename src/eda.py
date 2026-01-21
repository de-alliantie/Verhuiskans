"""You can run this file to generate an Exploratory Data Analysis report.

The program will likely not run without errors, but still create a report file (.html) in the /notebooks folder.
"""
from ydata_profiling import ProfileReport

from src.utils.io import LEVEL, generate_data_dir_path, load_from_pkl

# Load your DataFrame
df_combined_path = generate_data_dir_path(LEVEL.LOAD, "df_combined", suffix=".pickle")
df_combined = load_from_pkl(df_combined_path)

# Generate the report
profile = ProfileReport(df_combined)

# Save the report to an HTML file
profile.to_file("notebooks/df_combined_profile.html")
print("")
