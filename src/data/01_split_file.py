import pandas as pd


def split_csv(input_file, output_file_base, num_splits):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Calculate the number of rows per split
    num_rows = len(df)
    rows_per_split = num_rows // num_splits

    # Split the dataframe into smaller dataframes and save them as CSV files
    for i in range(num_splits):
        start_row = i * rows_per_split
        # Ensure the last split includes any remaining rows
        end_row = (i + 1) * rows_per_split if i != num_splits - 1 else num_rows
        df_split = df.iloc[start_row:end_row]

        # Write the split dataframe to a new CSV file
        output_file = f"{output_file_base}_{i+1}.csv"
        df_split.to_csv(output_file, index=False)
        print(f"Saved {output_file}")


# Example usage
input_file = "../../data/raw/HomeC.csv"
output_file_base = "../../data/raw/chunks/HomeC_split"
num_splits = 30

split_csv(input_file, output_file_base, num_splits)
