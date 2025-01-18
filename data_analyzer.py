import pandas as pd

label = 'postopmetastaz'
df = pd.read_excel(f"dataset/input_excels/{label}_cleared_data.xlsx")

print(df[label].value_counts())

