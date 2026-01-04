# Reading into data frame from file
import pandas as pd

data_frame = pd.read_csv("../data/dupes_dropped_creditcard.csv")

# Profiling dataset
from ydata_profiling import ProfileReport

profile = ProfileReport(data_frame, title="No Dupes Profiling Report")
profile.to_file("./no_dupe_data_report.html")
