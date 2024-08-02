import pandas as pd

drug1chem = pd.read_csv("drugs1Descriptors_filtered.csv", index_col=0)
drug1chem = drug1chem.reset_index(drop=True)

drug2chem = pd.read_csv("drugs2Descriptors_filtered.csv", index_col=0)
drug2chem = drug2chem.reset_index(drop=True)

drug1chem.to_csv('drug1_chem.csv',index=False, header=False)
drug2chem.to_csv('drug2_chem.csv',index=False, header=False)

gex = pd.read_csv("gex.csv", index_col=0)
gex.to_csv('cell_line_gex.csv',index=False, header=False)
