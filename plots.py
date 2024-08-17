import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 读取数据
gdsc = pd.read_csv('test_comb_data.csv')
gdsc['Bliss_matrix'] = gdsc['Bliss_matrix']*500
# 绘制分布图
mean_value = gdsc['Bliss_matrix'].mean()
median_value = gdsc['Bliss_matrix'].median()
min_value = gdsc['Bliss_matrix'].min()
max_value = gdsc['Bliss_matrix'].max()

plt.figure(figsize=(10, 6))
sns.kdeplot(gdsc['Bliss_matrix'], fill=True, color='skyblue', alpha=0.5)
plt.axvline(mean_value, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='g', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')

range_str = f'Min: {min_value:.2f}\nMax: {max_value:.2f}'
textstr = f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n{range_str}'

plt.xlim(-125, 100)
plt.ylim(0, 0.65)
plt.legend()
plt.xlabel('Synergy Bliss')
plt.ylabel('Density')
plt.xlim(-125, 100)
plt.ylim(0, 0.065)

plt.gca().annotate(textstr, xy=(0.81, 0.8), xycoords='axes fraction',
                    fontsize=12, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()

Comb = pd.read_csv('train_set/DrugCombinationData.tsv',sep='\t')

mean_value = Comb['synergy_loewe'].mean()
median_value = Comb['synergy_loewe'].median()
min_value = Comb['synergy_loewe'].min()
max_value = Comb['synergy_loewe'].max()

plt.figure(figsize=(10, 6))
sns.kdeplot(Comb['synergy_loewe'], fill=True, color='skyblue', alpha=0.5)
plt.axvline(mean_value, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='g', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')

range_str = f'Min: {min_value:.2f}\nMax: {max_value:.2f}'
textstr = f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\n{range_str}'

plt.xlim(-125, 100)
plt.ylim(0, 0.065)
plt.legend()
plt.xlabel('Synergy Loewe')
plt.ylabel('Density')

plt.gca().annotate(textstr, xy=(0.81, 0.8), xycoords='axes fraction',
                    fontsize=12, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.show()