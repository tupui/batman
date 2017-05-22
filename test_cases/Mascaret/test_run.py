from batman.functions import MascaretApi
from batman.functions.mascaret import print_statistics, histogram

# Create an instance of MascaretApi
study = MascaretApi('config_canal.json','config_canal_user.json')
#study = MascaretApi('config_garonne_lnhe.json','config_garonne_lnhe_user.json')

# Print informations concerning this study
print(study)

# Perform a specific study
#h = study([30, 3000])
#print('Water level computed with Ks = 30, Q = 3000',h)

# Plot the water level along the open-channel at final time
#study.plot_opt('ResultatsOpthyca.opt')

# Realize the study with the user defined tasks 
h = study()
print('Water level computed with json user defined values', h)
study.plot_opt('ResultatsOpthyca.opt')

# Print and plot statistics concerning the model output uncertainty
#print_statistics(h)
#histogram(h, xlab='Water level at Marmande', title='Distribution of the uncertainty')


# Details about MascaretApi
#help(MascaretApi)
