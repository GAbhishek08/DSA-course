import pandas as pd
from scipy import stats

df = pd.read_csv("C:/Users/Guguloth Abhishek/OneDrive/Desktop/DSA course/Assignment 5/Hip_star.csv")
hyades = df[(df['RA']<100) & (df['RA']>50) & (df['DE']<25) & (df['DE']>0) & (df['pmRA']<130) & (df['pmRA']>90) & (df['pmDE']<-10) & (df['pmDE']>-60)]
non_hyades = df[(df['RA']>100) | (df['RA']<50) | (df['DE']>25) | (df['DE']<0) | (df['pmRA']>130) | (df['pmRA']<90) | (df['pmDE']>-10) | (df['pmDE']<-60)]

hyades_BV_var = hyades['B-V'].var()
non_hyades_BV_var = non_hyades['B-V'].var()

print('Variance of B-V column of Hyades : ', hyades_BV_var )
print('Variance of B-V column of Non Hyades : ', non_hyades_BV_var )

div = non_hyades_BV_var/hyades_BV_var

# Standard Deviation of Non Hyades > Hyades
if div<4:
    print('We can proceed to perform the two sample t-test with equal variances since the ratio of variances of two samples is less than 4')
else:
    print("We need to perform Welchs t-test, which does not assume equal variances")

# Two sample ttest assuming equal variances
t_value, p_value = stats.ttest_ind(a=hyades['B-V'], b=non_hyades['B-V'], equal_var=True)
print("The t test statistic is {0} and the corresponding two-sided p-value is {1}".format(t_value, p_value))

if p_value>0.05:
    print("Null Hypothesis is not rejected and there is no difference between the color(B-V) of Hyades and Non Hyades stars")
else:
    print("Null Hypothesis is rejected and the color(B-V) of the Hyades stars differ from the Non Hyade ones")
