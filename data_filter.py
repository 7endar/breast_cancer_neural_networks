import pandas as pd

input_file = 'dataset/input_excels/filled_data.xlsx'  # original dataset

df = pd.read_excel(input_file)
label = 'neosonrasıpathboyutu'  # target label for the model

# columns to check for null data
columns_to_check = [
    'yas', 'neoboyutUSG', 'neooncesiaksilla', 'kliniklenfkat',
    'estkat', 'progkat2',
    'cerb2', 'ki67pre', 'preopmetastazvarlığı', 'KLİNİKLAP', 'USGLAP', 'MRLAP', 'ÇAP', 'SAYI', label
]

df_cleaned = df.dropna(subset=columns_to_check)  # new filtered dataset

output_file = f'dataset/input_excels/{label}_cleared_data.xlsx'  # save the new dataset with the name of the target label
df_cleaned.to_excel(output_file, index=False)

print(f"Rows with values in all specified columns were saved to the new file: {output_file}")
