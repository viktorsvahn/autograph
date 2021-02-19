#!/usr/bin/env pythpn3.5

import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


###########################################################################################################################################
##  README:                                                                                                                              ##
## -Only the premable needs to be filled out. The program automatically outputs a fitted curve according to the chosen type and values.  ##
##                                                                                                                                       ##
## -All plotted files must be present in the same directory which must also be specified, unless the data is in the same folder as the   ##
##  script itself. The same goes for saving the figure.                                                                                  ##
##                                                                                                                                       ##
## -Filenames are specified within the file list.                                                                                        ##
##                                                                                                                                       ##
## -The script supports linear regression and mono-exponential fitting.                                                                  ##
##                                                                                                                                       ##
##  Viktor Svahn, 2021-02-18                                                                                                             ##
###########################################################################################################################################

##############
## PREAMBLE ##
##############

# Specify title and axis labels.
plot_title = 'Title'
x_axis_label = 'x'
y_axis_label = 'y'

# Set type of curve fitting and alpha on scatter points
curve_fit_type = 'mono-exp' # Choose from: none, lin-reg, mono-exp.
scatter_alpha_amount = 0.3 # Sets transparency of scatter points.

# Output file specifics such as file- type, name, location.
show_plot = True
savefile = False
save_name = ''
save_location = ''

# Input file location and filnames. Files takes a list of filenames with number of rows to skip in each file.
# The skip functionality is used to ignore non-data rows of output files from instruments an such.
# Ex. files = [ ['filename1', skiprows], ['filename2', skiprows] ]
file_location = 'Data/'
files = [
	['exp_data', 0],
	['exp_data2', 0]
]


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
scatter_point_type ='.' # See matplotlib documentation for variants, e.g. 'o', 's' etc.
legend_position = '' # Set legend position. If empty 'best' is chosen.


# Axis scaling factor. Affects all input data equally.
# Scales data according to given factors
x_axis_factor = 1
y_axis_factor = 1


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################


# Handling of file list using dataframes. The use of data frames helps indexing while keeping
# the structure of the file list above readable. This way the number of rows to skip is more
# clearly associated with a specific file.
files_df = pd.DataFrame(files, columns = ['Filename', 'Skiprows'])
files_index = files_df.index
print('Input:')
print(files_df)
print('Make sure the correct number of lines are skipped in each data file.')


################
## FORMATTING ##
################

# Sets style according to whats set in the preamble.
if use_style != 'standard':
	if use_style == 'latex':
		matplotlib.rcParams['mathtext.fontset'] = 'stix'
		matplotlib.rcParams['font.family'] = 'STIXGeneral'
		matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
	elif use_style == 'latex-math':
		matplotlib.rcParams['mathtext.fontset'] = 'stix'
		matplotlib.rcParams['font.family'] = 'STIXGeneral'
		matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

###############
## FUNCTIONS ##
###############


def array_from_file(file, column, rowskip):
	"""
	Outputs loaded values (ndarray) from a desired column of a given file. 
	"""
	file_data = np.loadtxt(file, skiprows=rowskip, dtype=np.float64)
	column_data = file_data[:,column]
	return column_data

def find_nearest(array, value):
	"""
	Finds nearest value in a given array. 

	Used to find actual data points in an array close to some value of interest.
	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]

def exponential(x, a, b, c):
	"""
	Defines an exponential functions to be fit to data points.
	"""
	return a * np.exp(b * x) + c

def linear_fit(x,y):
	"""
	Applies a linear regression to x,y using scipy.stats.linregress.
	"""
	return stats.linregress(x,y)

def mono_exp_fit(x,y):
	"""
	Applies a mono-exponential fit to x,y. The fitting parameters are guessed automatically.

	The slope of a mono-exponential is given by slope = -ln(2)/x_half, where x_half denotes
	the position along the x-axis where y is at half maximum.

	To estimate the error, the mean value is taken over the first or last 2% elements, rounded 
	to the nearest integer, of the number of points in y. The first values are chosen if the
	exponential is of growth type, otherwise the last values or chosen.
	

	The starting index is chosen as the absolute maximum value of the y-axis. If the value of
	y = y/2 is positive this value is max(y) whereas a negative value means it is min(y). 
	These are, respectively, the intercepts as well.
	"""

	
	# Slope estimation. Index of mid y-value is found. 
	y_half = find_nearest(y, ( max(y) + min(y) )/2 )
	y_half_id = np.where( y == y_half )
	x_at_half_y = x[ y_half_id ]
	slope_guess = -float( np.log(2)/x_at_half_y )
	
	# Number of values error_guess is based on.
	error_lower_bound_index = math.ceil(len(y)*0.02)

	if slope_guess > 0:
		error_guess = np.mean( y[:error_lower_bound_index] )
		x_zero = find_nearest(x, 0)
		x_zero_index = np.where( x == x_zero )
		intercept_guess = float( y[ x_zero_index ])
		result = [ intercept_guess, slope_guess, error_guess ]
		print(result)
	else:
		error_guess = np.mean( y[-error_lower_bound_index:] )
		if y_half > 0:
			intercept_guess = max(y)
		else:
			intercept_guess = min(y)

	# Curve fitting.
	popt, pcov = curve_fit(
		exponential, 
		x[start_index:, ], 
		y[start_index:, ], 
		p0=(
			intercept_guess, 
			slope_guess, 
			error_guess
			) 
		)
	return popt, pcov


##########
## PLOT ##
##########

output = []

# For loop that iterates and plots over all specfied files.
for i in files_index:
	filename = files[i][0]
	skiprows = files[i][1]

	x_values = array_from_file(
		file_location+filename,
		0,
		skiprows
		)*x_axis_factor
	y_values = array_from_file(
		file_location+filename, 
		1, 
		skiprows
		)*y_axis_factor

	if curve_fit_type == 'none':
		plt.plot(
			x_values, 
			y_values, 
			'.', 
			label=filename,
			alpha=scatter_alpha_amount
			)

	# Curve fiting.
	elif curve_fit_type != 'none':
		plt.plot(
			x_values, 
			y_values, 
			scatter_point_type, 
			label=filename, 
			alpha=scatter_alpha_amount
			)
		x = np.linspace(
			min(x_values), 
			max(x_values), 
			len(x_values)*1e3
			)

		# Plots linear regression
		if curve_fit_type == 'lin-reg':
			res = linear_fit(x_values, y_values)

			# Prepare result ouput.
			result = [ filename, res.intercept, res.slope, res.rvalue**2 ]
			output.append(result)

			plt.plot(
				x, 
				res.intercept + res.slope*x, 
				label='$y= %6.5fx + %6.5f, \\quad R^2=%6.5f $' % (+round(res.slope, 4), round(res.intercept, 4), round(res.rvalue**2, 5)))

		# Plots mono-exponential fit. Asborpion or emission type is detected by the integral sign.
		elif curve_fit_type == 'mono-exp':
			if np.trapz(y_values, x=x_values) > 0:
				start_index = np.argmax(y_values)
			else:
				start_index = np.argmin(y_values)
			popt, pcov = mono_exp_fit(x_values, y_values)

			# Prepare result ouput.
			result = [ filename, popt[0], popt[1], popt[2] ]
			output.append(result)			

			plt.plot(
				x_values[start_index:, ], 
				exponential(x_values[start_index:, ], 
				*popt), 
				label='$y= %6.5E e^{ %6.5E t} %+6.5E $' % tuple(popt) 
				) 
	
# Results output in a DataFrame environment.

if curve_fit_type == 'lin-reg':
	output_df = pd.DataFrame(output, columns = [ 'Filename', 'Intercept', 'Slope', 'R-squared'] )
	print('Output:')
	print(output_df)
elif curve_fit_type == 'mono-exp':		
	output_df = pd.DataFrame(output, columns = [ 'Filename', 'Intercept', 'Slope', 'Error'] )
	print('Output:')
	print(output_df)

	
if set_fontsize:
	plt.title(plot_title, fontsize=title_fontsize)
	plt.xlabel(x_axis_label, fontsize=axis_fontsize)
	plt.ylabel(y_axis_label, fontsize=axis_fontsize)
	plt.tick_params(labelsize=axis_ticksize)
else:
	plt.title(plot_title)
	plt.xlabel(x_axis_label)
	plt.ylabel(y_axis_label)

if legend_position == '':
	plt.legend(loc='best')
else:
	plt.legend(loc=legend_position)

if savefile:
	plt.savefig(
		save_location+save_name+'.'+figure_format, 
		dpi=resolution_dpi, 
		format=figure_format
		)

if show_plot:
	plt.show()