import numpy as np
from scipy import stats




def test_stat(stat_func, alpha, data, data_name):

	print()
	print(stat_func.__name__,'alpha=%.3f'%alpha,'-->', data_name)
	statistic, p = stat_func(*data)
	print('statistic =', statistic)
	print('pvalue =', p)
	if p < alpha: # null hypothesis: x comes from a normal distribution
		print('The null hypothesis can be rejected')
		return 'The null hypothesis can be rejected'
	else:
		print('The null hypothesis cannot be rejected')
		return "The null hypothesis cannot be rejected"

def my_test():
	print('null = normal distribution')
	dump63 = [80.35, 81.41, 80.85, 85.96, 81.58, 80.07, 81.58, 83.16, 83.44, 82.82, ]
	rel_index0 = [ 83.60, 85.68, 83.49, 87.09, 85.01, 82.54, 85.79, 85.18, 80.69, 86.08, 85.12, ]
	dump63 = np.array(dump63)
	rel_index0 = np.array(rel_index0)
	
	alpha = 0.05
	test_stat(stats.shapiro, alpha, [dump63], 'dump63')
	test_stat(stats.shapiro, alpha, [rel_index0], 'rel_index0')
	test_stat(stats.ttest_ind, alpha, [rel_index0, dump63], 'dump63 and rel_index0')

if __name__ == '__main__':
	my_test()