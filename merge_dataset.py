import pandas as pd

# load original lithium dataset
main_df = pd.read_csv(
    r"C:\Users\HP\Downloads\Lithium_Statistics (1).csv"
)

# load terrain variables
terrain_df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\VS_Code\terrain_variables.csv"
)

# merge both datasets
merged_df = pd.merge(
    main_df,
    terrain_df,
    on=["lat","long"],
    how="left"
)

# save final dataset
merged_df.to_csv(
    r"C:\Users\HP\OneDrive\Desktop\VS_Code\Lithium_with_DEM.csv",
    index=False
)

print("Merged dataset created successfully")