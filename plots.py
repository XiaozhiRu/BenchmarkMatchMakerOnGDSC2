import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 读取数据
gdsc = pd.read_csv('DrugCombinationData_Bliss.csv',index_col=0)

# 绘制分布图
plt.figure(figsize=(10, 6))
sns.histplot(gdsc['synergy_bliss'], bins=30, kde=True)
plt.title('Distribution of Synergy Bliss Scores')
plt.xlabel('Synergy Bliss Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()