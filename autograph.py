#!/usr/bin/env pythpn3.5

import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy import stats
from scipy.optimize import curve_fit
from cycler import *
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# Handling of file list using dataframes. The use of data frames helps indexing while keeping
# the structure of the file list above readable. This way the number of rows to skip is more
# clearly associated with a specific file.

files_df = pd.DataFrame(files, columns = ['Filename', 'Skiprows', 'x-col.', 'y-col.', 'x-scale', 'y-scale', 'Marker', 'Label'])
files_index = files_df.index
print('\n---------------------------------------------------------------------')
print('Input:')
print('---------------------------------------------------------------------')
print(files_df)
print('---------------------------------------------------------------------')
print('Make sure the correct number of lines are skipped and that the columns\nin each data file match.\n\n')

################
## FORMATTING ##
################

# Sets style according to whats set in the preamble.
if use_style != 'standard':
	if use_style == 'latex':
		mpl.rcParams['mathtext.fontset'] = 'stix'
		mpl.rcParams['font.family'] = 'STIXGeneral'
		#plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
	elif use_style == 'latex-math':
		mpl.rcParams['mathtext.fontset'] = 'stix'
		mpl.rcParams['font.family'] = 'STIXGeneral'
		#plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


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

def mono_exponential(x, a, b, c):
	"""
	Defines an exponential functions to be fit to data points.
	"""
	return a * np.exp(-b * x) + c

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
	slope_guess = float( np.log(2)/x_at_half_y )
	

	# Number of values error_guess is based on.
	error_lower_bound_index = math.ceil(len(y)*0.02)

	if slope_guess < 0:
		start_index = 0
		error_guess = np.mean( y[:error_lower_bound_index] )
		x_zero = find_nearest(x, 0)
		x_zero_index = np.where( x == x_zero )
		intercept_guess = float( y[ x_zero_index ])
		#result = [ intercept_guess, slope_guess, error_guess ]
		#print(result)
	else:
		error_guess = np.mean( y[-error_lower_bound_index:] )
		if y_half > 0:
			start_index = np.argmax(y)
			intercept_guess = max(y)
		else:
			start_index = np.argmin(y)
			intercept_guess = min(y)

	# Curve fitting.
	popt, pcov = curve_fit(
		mono_exponential, 
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

fig = plt.figure(figsize=figure_size)
ax = plt.axes()

# Defines colormaps used in the plot
if colormap == '':
	None
elif colormap == 'viridis':
	colors = pl.cm.viridis(np.linspace(0,1, len(files)))
	ax.set_prop_cycle(cycler('color', colors) )

elif colormap == 'magma':
	colors = pl.cm.magma(np.linspace(0,1, len(files)))
	ax.set_prop_cycle(cycler('color', colors) )
elif colormap == 'plasma':
	colors = pl.cm.plasma(np.linspace(0,1, len(files)))
	ax.set_prop_cycle(cycler('color', colors) )
elif colormap == 'inferno':
	colors = pl.cm.inferno(np.linspace(0,1, len(files)))
	ax.set_prop_cycle(cycler('color', colors) )

output = []


if normalize_data_auc and normalize_data_max_value:
	print('Only one type of normalization is allowed.')

else:
	# For loop that iterates and plots over all specfied files.
	for i in files_index:
		filename = files[i][0]
		skiprows = files[i][1]
		x_axis = files[i][2]
		y_axis = files[i][3]
		scale_x = files[i][4]
		scale_y = files[i][5]
		if files[i][6] == '':
			marker_type = scatter_point_type
		else:
			marker_type = files[i][6]
		if files[i][7] == '':
			plot_label = filename
		else:
			plot_label = files[i][7]


		x_values = array_from_file(
			file_location+filename,
			x_axis,
			skiprows
			)*scale_x
		if x_log_scale[0]:
			if x_log_scale[1] == '2' or 2:
				x_values = np.log2(x_values)
			if x_log_scale[1] == 'e':
				x_values = np.log(x_values)
			elif x_log_scale[1] == '10' or 10:
				x_values = np.log10(x_values)
			else:
				print('Unsupported base.')

		y_values = array_from_file(
			file_location+filename, 
			y_axis, 
			skiprows
			)*scale_y
		if y_log_scale[0]:
			if y_log_scale[1] == '2' or 2:
				y_values = np.log2(y_values)
			if y_log_scale[1] == 'e':
				y_values = np.log(y_values)
			elif y_log_scale[1] == '10' or 10:
				y_values = np.log10(y_values)
			else:
				print('Unsupported logarithmic base. Log scale must be natural, 2 or 10.')

		if normalize_data_auc:
			normalization_factor = np.trapz(y_values, x=x_values)
			y_values = y_values/abs( normalization_factor )
		elif normalize_data_max_value:
			normalization_factor = max(y_values)
			y_values = y_values/normalization_factor


		if curve_fit_type == '':
			ax.plot(
				x_values, 
				y_values, 
				marker_type,
				label=plot_label,
				alpha=scatter_alpha_amount,
				markevery=mark_every
				)

		# Curve fitting.
		else:
			plt.plot(
				x_values, 
				y_values, 
				marker_type,
				label=plot_label,
				alpha=scatter_alpha_amount,
				markevery=mark_every
				)
			x = np.linspace(
				min(x_values), 
				max(x_values), 
				len(x_values)*1e2
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
					label='$y= {:E} x {:+E}, \\quad R^2={:f} $'.format(res.slope, res.intercept, res.rvalue**2),
					linewidth=curve_fit_linewidth
					)

			# Plots mono-exponential fit. Asborpion or emission type is detected by the integral sign.
			elif curve_fit_type == 'mono-exp':
				popt, pcov = mono_exp_fit(x_values, y_values)

				slope = popt[1]
				y_half = find_nearest(y_values, ( max(y_values) + min(y_values) )/2 )

				if slope < 0:
					start_index = x_values[0]
				else:
					if y_half > 0:
						start_index = np.argmax(y_values)
					else:
						start_index = np.argmin(y_values)


				#print(pcov)
				# Prepare result ouput.
				result = [ filename, popt[0], -popt[1], popt[2] ]
				output.append(result)			

				plt.plot(
					x_values[start_index:, ], 
					mono_exponential(x_values[start_index:, ], 
					*popt),
					label='$y= %E \\times e^{ %E t} %+E $' % (popt[0], -popt[1], popt[2]),
					linewidth=curve_fit_linewidth
					) 
		
	# Results output in a DataFrame environment.
	if curve_fit_type != '':
		if curve_fit_type == 'lin-reg':
			output_df = pd.DataFrame(
				output, 
				columns = [ 
				'Filename', 
				'Intercept', 
				'Slope', 
				'R-squared'
				] )

		elif curve_fit_type == 'mono-exp':		
			output_df = pd.DataFrame(
				output, 
				columns = [
				 'Filename', 
				 'Intercept', 
				 'Slope', 
				 'Error'
				 ] )

		pd.set_option('display.float_format', lambda x: '%.5E' % x)
		print('\n---------------------------------------------------------------------')
		print('Output:')
		print('---------------------------------------------------------------------')
		print(output_df)
		print('---------------------------------------------------------------------')


	# Plot settings, based on premable.
	ax.set_title(
		plot_title, 
		fontsize=title_fontsize
		)
	ax.set_xlabel(
		x_axis_label, 
		fontsize=axis_fontsize
		)
	ax.set_ylabel(
		y_axis_label, 
		fontsize=axis_fontsize
		)
	ax.tick_params(
		labelsize=axis_ticksize,
		direction='tick_direction',
		length=axis_ticksize*0.35, 
		bottom=True,
		top=False, 
		left=True,
		right=False
		)
	ax.set_xlim(x_axis_limits)
	ax.set_ylim(y_axis_limits)


	if legend_position == '':
		ax.legend(loc='best', shadow=legend_shadow)
	else:
		ax.legend(loc=legend_position, shadow=legend_shadow)

	if savefile:
		plt.savefig(
			save_location+save_name+'.'+figure_format, 
			dpi=resolution_dpi, 
			format=figure_format
			)

	if show_plot:
		plt.show()
