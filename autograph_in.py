##############
## PREAMBLE ##
##############

# Enter path to the script autograph.py. Path must end with a slash ('/').
path_to_script = ''

# Input file location and filnames. Files takes a list of filenames with number of rows to skip in each file.
# The skip functionality is used to ignore non-data rows of output files from instruments an such.
file_location = 'Data/'

# Ex. files = [ ['filename1', skiprows, x-column, y-column, scale factor x-axis, scale factor y-axis],
#				 ['filename2', skiprows, x-column, y-column, scale factor x-axis, scale factor y-axis] ]
# For regular plotting, set scaling to 1 on all axes.
files = [
	#['filename1', skiprows, x-column, y-column, scale factor x-axis, scale factor y-axis, alternative marker type]
	['exp_data', 0, 0, 1, 1, 1, ''],
]


# Specify title and axis labels.
plot_title = 'Title'
x_axis_label = 'x'
y_axis_label = 'y'

# Set type of curve fitting and alpha on scatter points
curve_fit_type = '' # Choose from: lin-reg, mono-exp. Leave empty '' if no fitting.
curve_fit_linewidth = 2 # Sets linewidth of fited curve.
scatter_alpha_amount = 0.5	# # Sets transparency of scatter points.

# Output file specifics such as file- type, name, location.
show_plot = True
savefile = False
save_name = ''
save_location = ''


#######################
## ADVANCED SETTINGS ##
#######################

# Figure style. Choose between; standard, latex, latex-math.
use_style = 'latex'

# Input output file specifics such as file- type, name, resoltion.
resolution_dpi = 300
figure_format = 'eps'

# Plot specifics, fontsize, ticksize, scatter poit--type, legend position other than 'best'.
set_fontsize = False
title_fontsize = 18
axis_fontsize = 16
axis_ticksize = 14
tick_direction = 'in' # Choose from in, out and inout. 
scatter_point_type ='.' # See matplotlib documentation for variants, e.g. 'o', 's' etc.
mark_every =  None # For 'mark_every=N' the each plot contains every Nth point in a scatter plot. Set None to mark all points.
legend_position = '' # Set legend position. If empty 'best' is chosen.
legend_shadow = True

# Sets log-scale on specified axis by setting the bool-of-choice in the tuples. Choose from bases 2, e and 10 and enter either str or int.
x_log_scale = (False, 'e')
y_log_scale = (False, '10')

# Sets limits to axes to given lists or tuples.
x_axis_limits = None
y_axis_limits = None

# Normalization of data by are-under-curve or maximum value.
normalize_data_max_value = False
normalize_data_auc = False


exec(open(path_to_script+'autograph.py').read())
