import pandas as pd
import openpyxl

# Define the path to the uploaded text file
file_path = 'obqaformat.txt'

# Initialize lists to store each column of data
model_names, datasets, forms, max_token_lengths, min_token_lengths, avg_token_lengths, final_accuracies = [], [], [], [], [], [], []

# Read and parse the text file line by line
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("模型名称:"):
            model_names.append(line.split(": ")[1])
        elif line.startswith("数据集:"):
            datasets.append(line.split(": ")[1])
        elif line.startswith("Form:"):
            forms.append(line.split(": ")[1])
        # elif line.startswith("Max Token Length:"):
        #     max_token_lengths.append(line.split(": ")[1])
        # elif line.startswith("Min Token Length:"):
        #     min_token_lengths.append(line.split(": ")[1])
        elif line.startswith("Avg Token Length:"):
            avg_token_lengths.append(line.split(": ")[1])
        elif line.startswith("Final Accuracy:"):
            final_accuracies.append(line.split(": ")[1])

# Create a DataFrame from the parsed lists
df = pd.DataFrame({
    "Model Name": model_names,
    "Dataset": datasets,
    "Form": forms,
    # "Max Token Length": max_token_lengths,
    # "Min Token Length": min_token_lengths,
    "Avg Token Length": avg_token_lengths,
    "Final Accuracy": final_accuracies
})

# Extract main model category and sub-model name for hierarchical grouping
df['Model Category'] = df['Model Name'].apply(lambda x: x.split('/')[0])
df['Sub-model'] = df['Model Name'].apply(lambda x: x.split('/')[1])

# Sort the DataFrame by "Model Category" and then "Sub-model" for hierarchical structure
df = df.sort_values(by=["Model Category", "Sub-model"]).reset_index(drop=True)

# Save the DataFrame to an Excel file
output_path_hierarchical = 'obqaformat_hierarchical.xlsx'
# df.to_excel(output_path_hierarchical, index=False)

# Load the saved Excel file and enforce numeric formatting where possible
workbook = openpyxl.load_workbook(output_path_hierarchical)
sheet = workbook.active

# Define cell formatting for all cells to be numeric if possible
for row in sheet.iter_rows():
    for cell in row:
        try:
            # Attempt to convert cell value to a float to enforce numeric formatting
            cell.value = float(cell.value)
        except (ValueError, TypeError):
            # If conversion fails (for headers or non-numeric data), leave the cell as-is
            pass

# Save the workbook with numeric formatting
output_path_numeric_format = 'obqaformat_numeric.xlsx'
workbook.save(output_path_numeric_format)

# Output the path of the final saved file
output_path_numeric_format
