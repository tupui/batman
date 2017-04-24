from batman.functions import MascaretApi
from batman.functions.mascaret import print_statistics, histogram

# Create an instance of MascaretApi
study = MascaretApi('config_garonne_lnhe.json','config_garonne_lnhe_user.json')

# Print informations concerning this study
print(study)

# Perform a specific study
h = study([30, 3000])
print('test1',h)

# Plot the water level along the open-channel at final time
study.plot_opt('ResultatsOpthyca.opt')

# Realize the study with the user defined tasks 
#h = study()
#print('test2',h)

# Print and plot statistics concerning the model output uncertainty
#print_statistics(h)
#histogram(h, xlab='Water level at Marmande', title='Distribution of the uncertainty')


# Details about MascaretApi
#help(MascaretApi)
