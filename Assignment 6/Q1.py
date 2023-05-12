from scipy import stats

# Given values
eins_predic = 1.74
newton_predic = 0.87
edd_val = 1.61
edd_err = 0.4
cromm_val = 1.98
cromm_err = 0.16

# Calculating bayes factor
a = stats.norm.pdf(edd_val, loc=eins_predic, scale=edd_err)
b = stats.norm.pdf(edd_val, loc=newton_predic, scale=edd_err)

a1 = stats.norm.pdf(cromm_val, loc=eins_predic, scale=cromm_err)
b1 = stats.norm.pdf(cromm_val, loc=newton_predic, scale=cromm_err)

bayes_factor_edd = a/b
bayes_factor_cromm = a1/b1

print("Bayes Factor for Eddington Value :", bayes_factor_edd)
print("Bayes Factor for Crommelin Value :", bayes_factor_cromm)