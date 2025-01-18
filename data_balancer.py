import pandas as pd
from sklearn.utils import resample

label = 'neosonrasıpathboyutu'  # desired label
count = 73  # desired (or minimum) data count for each class

df = pd.read_excel(f"dataset/input_excels/{label}_balanced_data.xlsx")

output_path = f"dataset/input_excels/{label}_balanced_data.xlsx"

balanced_data = []

for class_label, group in df.groupby(label):
    if len(group) >= count:  # enough data, select randomly
        sampled = resample(group, replace=False, n_samples=count, random_state=7)
    else:
        # not enough data, add the all available data
        all_samples = group.copy()
        # random sampling to fill in the gaps
        missing_count = count - len(all_samples)
        additional_samples = resample(
            group, replace=True, n_samples=missing_count, random_state=7
        )
        sampled = pd.concat([all_samples, additional_samples])

    balanced_data.append(sampled)

balanced_df = pd.concat(balanced_data).reset_index(drop=True)
balanced_df.to_excel(output_path, index=False)

print(f"All classes in the {label} were balanced to have {count} instances each and saved to: {output_path}.")