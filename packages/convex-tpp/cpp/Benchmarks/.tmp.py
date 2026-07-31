
import os
import pandas as pd

for file in os.listdir(".."):
	if not file.endswith(".csv"):
		continue
	
	df = pd.read_csv(file)
	other = pd.read_csv("../" + file)

	column = other["Binary Search"]
	column.name = "Binary Search (lazy)"
	
	df = pd.concat([df, column], axis=1)
	df.to_csv(file, index=False)