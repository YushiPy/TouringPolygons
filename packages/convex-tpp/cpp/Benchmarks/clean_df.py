
import os

import pandas as pd

def clean_df(df: pd.DataFrame) -> pd.DataFrame:

	times = [df[x] for x in df.columns if x not in {'k', 'n'}]

	# Remove outliers
	for timings in times:

		# Get the 95th percentile of the timings
		percentile_95 = timings.quantile(0.95)

		values = timings.values

		for i in range(1, len(values) - 1):
			if values[i] > 2 * percentile_95:
				values[i] = (values[i - 1] + values[i + 1]) / 2

		# If the first timing is an outlier, replace it with the second timing
		if values[0] > 2 * percentile_95:
			values[0] = values[1]
		
		# If the last timing is an outlier, replace it with the second to last timing
		if values[len(values) - 1] > 2 * percentile_95:
			values[len(values) - 1] = values[len(values) - 2]

		# Update the timings in the dataframe
		df.loc[:, timings.name] = values

	return df

if __name__ == "__main__":

	for filename in os.listdir("."):
		if filename.endswith(".csv"):
			df = pd.read_csv(filename)
			df = clean_df(df)
			df.to_csv(filename, index=False)
