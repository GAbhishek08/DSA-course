from scipy import stats

# Calculationg significance value of atlas
p_val1 = 1.7 * (10**(-9))
sig_atlas = stats.norm.isf(p_val1)
print("The significance value of the ATLAS discovery paper = ",sig_atlas)

# Calculating significance value of ligo
p_val2 = 2 * (10**(-7))
sig_ligo = stats.norm.isf(p_val2)
print("The significance value of the LIGO discovery paper = ",sig_ligo)

# Calculating Chi square GOF value of Super-K Discovery paper
dof = 67  # No. of Degrees Of Freedom
chi2_min = 65.2
chi2_gof = 1 - stats.chi2(dof).cdf(chi2_min)
print("The chi square GOF value of Super-K Discovery paper = ", chi2_gof)