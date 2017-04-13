from batman.functions import MascaretApi
from batman.functions.mascaret import print_statistics, histogram

# Create an instance of MascaretApi
study = MascaretApi('config_garonne_lnhe.json','config_garonne_lnhe_user.json')

# Print informations concerning this study
print(study)

# Run Mascaret with the user defined parameters
h = study.run_mascaret()
print(h)

# Realize the study with the user defined tasks (e.g. Monte-Carlo)
h = study()
print(h)

# Print and plot statistics concerning the model output uncertainty
print_statistics(h)
histogram(h, xlab='Water level at Marmande', title='Distribution of the uncertainty')

# Plot the water level along the open-channel at final time
study.plot_opt()

# Details about MascaretApi
#help(MascaretApi)
