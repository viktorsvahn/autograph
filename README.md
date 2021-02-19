# autograph
Automatically plots data from a given file. Settings such as style, curve fitting, sizes, save location, file format etc are set in the preamble.


# Features:
-Only the premable needs to be filled out. The program automatically outputs a fitted curve according to the chosen type and values.

-All plotted files must be present in the same directory which must also be specified, unless the data is in the same folder as the script itself. The same goes for saving the figure.

-Filenames are specified within the file list.

-The script supports linear regression and mono-exponential fitting.

# HOW_TO
Simplest version
1. Add the file autograph.py to a preferred folder. To keep track of different plots it is wise to change the name of the script to easily differentiate between plots.
2. Set data folder location by changing 'file_location'.
3. Set load the correct data files in 'files'. Each file is entered within a list of the form: ['filename1', skiprows, x-column, y-column, scale factor x-axis, scale factor y-axis]. Skiprows refer to the number of non-data rows and the columns are specified such as column0=0, column1=1 etc. An externally determined scale factor can be used if needed.
4.  Set title and axis labels.
5.  Set preferred 'curve_fit_type' by chosing 'lin-reg' (linear regression) or 'mono-exp' (mono exponential fitting). Leave blank otherwise.
6.  Set amount of transparency of scatter points by setting 'scatter_aplha_amount' to a preferred value. 
7.  Set 'show_plot = True' if a plot should pop up when running the script.
8.  Set 'savefile = True' and add a 'save_name' if a plot should be saved to the specified 'save_location'.
9.  Run the script.
 
