import pandas as pd

gdsc = pd.read_csv('gdsc.csv', index_col=0)
drug_comb_data = pd.read_csv('DrugCombinationData.tsv',sep='\t')

gdsc['DrugCombData_Index'] = gdsc['CELL_LINE_NAME'].apply(
    lambda x: drug_comb_data[drug_comb_data['cell_line_name'] == x].index[0] if any(drug_comb_data['cell_line_name'] == x) else None
)

# 删除DrugCombData_Index为None的行
gdsc_filtered = gdsc.dropna(subset=['DrugCombData_Index'])

# 将DrugCombData_Index转换为整数
gdsc_filtered['DrugCombData_Index'] = gdsc_filtered['DrugCombData_Index'].astype(int)

# 保留gdsc数据对应的行，并选取lib1_name, lib2_name, CELL_LINE_NAME, Bliss_matrix四列
selected_columns = ['lib1_name', 'lib2_name', 'CELL_LINE_NAME', 'Bliss_matrix']
test_comb_data = gdsc_filtered.loc[:, selected_columns]

# 输出新的数据集test_comb_data
test_comb_data.to_csv('test_comb_data.csv', index=False)
gdsc_filtered.to_csv('gdsc_filtered.csv', index=False)

# 读取cell_line_gex.csv文件
cell_line_gex = pd.read_csv('cell_line_gex.csv',header=None)

# 初始化一个新的DataFrame来存储从cell_line_gex中获取的数据
test_cell_line = pd.DataFrame()

# 根据DrugCombData_Index填充cell_line_gex中的对应行
for idx in gdsc_filtered['DrugCombData_Index']:
    test_cell_line = test_cell_line.append(cell_line_gex.loc[idx], ignore_index=True)

test_cell_line.to_csv('test_cell_line.csv', index=False)
