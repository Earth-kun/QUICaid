input_file = 'C:/Users/Carlo Alamani/Desktop/code/QUICaid/tuning/firestuning.out'
output_file = 'firestuning-cleaned3.out'

# Define patterns to remove
patterns_to_remove = [
    "/home/carlo_alamani/.local/lib/python3.8/site-packages/sklearn/preprocessing/_data.py:462: RuntimeWarning: All-NaN slice encountered",
    "/home/carlo_alamani/.local/lib/python3.8/site-packages/sklearn/preprocessing/_data.py:461: RuntimeWarning: All-NaN slice encountered",
    "data_max = np.nanmax(X, axis=0)",
    "data_min = np.nanmin(X, axis=0)"
]

# Read the file and filter out unwanted lines
with open(input_file, 'r') as infile:
    lines = infile.readlines()

cleaned_lines = [line for line in lines if not any(pattern in line for pattern in patterns_to_remove)]

# Write the cleaned data to a new file
with open(output_file, 'w') as outfile:
    outfile.writelines(cleaned_lines)

print(f"Cleaned output written to {output_file}")
