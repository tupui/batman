from TelApy.tools.studyMASC_UQ import MascaretStudy

#MascaretStudy documentation
help(MascaretStudy)

# Create an instance of MascaretStudy from a JSON setting file.
study = MascaretStudy('config_Garonne.json', log_lvl='INFO', iprint = 1,working_directory = 'study_test')

# Print information concerning this study.
print(study)

# OPTION 1 : Run the instance of Mascaret for input data in JSON setting file.
hydraulic_state = study()

# POST-TREATMENT
# Extract the curvilinear abscissa and the hydraulic state
curv_abs = hydraulic_state['s']
water_depth = hydraulic_state['h']
water_elevation = hydraulic_state['z']
discharge = hydraulic_state['q']

print("[RESULTS] x = Curvilinear abscissa # z = Water elevation # q = Discharge")
for i, _ in enumerate(curv_abs):
    print("x = {} # z = {} # q = {}".format(curv_abs[i],
                                            water_elevation[i],
                                            discharge[i]))
# Plot the water elevation
study.plot_water(output="WaterElevation_fromJSON")
print("\n[PLOT] You can open the file 'WaterElevation_from_JSON.pdf' which contains"
      " a plot of the water elevation at final time.")
# Plot the bathymetry
study.plot_bathymetry(output="Bathymetry_fromJSON")
print("\n[PLOT] You can open the file 'Bathymetry_from_JSON.pdf' which contains"
      " a plot of the bathymetry at final time.")
# Save output files 
study.save(out_name = 'test_run')

# OPTION 2 : Run the instance of Mascaret for python defined input data 
Ks = [{'type': 'zone', 'index': 2, 'value': 30.0}]
Q = [{'type': 'discharge', 'index': 0, 'value': [3000.0, 3200.0]}]
hydraulic_state = study(x={'friction_coefficients': Ks,
                           'boundary_conditions': Q})
print(study)

# POST-TREATMENT
# Extract the curvilinear abscissa and the water level
curv_abs = hydraulic_state['s']
water_depth = hydraulic_state['h']
water_elevation = hydraulic_state['z']
discharge = hydraulic_state['q']
#
print("[RESULTS] x = Curvilinear abscissa # h = Water Depth # q = Discharge")
for i, _ in enumerate(curv_abs):
    print("x = {} # z = {} # q = {}".format(curv_abs[i],
                                            water_elevation[i],
                                           discharge[i]))

# Plot the water elevation
study.plot_water(output="WaterElevation_from_script")
print("\n[PLOT] You can open the file 'WaterElevation_from_script.pdf' which"
      " contains a plot of the water elevation at final time.")
# Plot the bathymetry
study.plot_bathymetry(output="Bathymetry_from_script")
print("\n[PLOT] You can open the file 'Bathymetry_from_script.pdf' which contains"
      " a plot of the bathymetry level at final time.")
# Save output files and del study
study.save(out_name = 'test_run')
del study

