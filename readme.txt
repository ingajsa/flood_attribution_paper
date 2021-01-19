
README 

flood_attribution_paper
code collection for the paper 10.21203/rs.3.rs-37259/v1

-----------------------------------------------------------------------------------------------------------

SYSTEM REQUIREMENTS

Python (3.6+) version of CLIMADA release v1.5.1 or later

For pre- and post-processing we also recommend Python 3

Tested on Python 3.7

Required non-standard python packages for post-processing are:

pyts.decomposition.SingularSpectrumAnalysis
pymannkendall
statsmodels
itertools
astropy.convolution

For the demo, we strongly recommend to install Jupyter-Notebook

2 INSTALLATION GUIDE

A detailed description on how to install CLIMADA is provided under

https://github.com/CLIMADA-project/climada_python

Typical installation time including all tests (~1.5h)

Post-processing (Python 3) can be done with any Python environment. 

3 DEMO

For damage generation with CLIMADA please see the RiverFlood Tutorial

https://github.com/CLIMADA-project/climada_python/blob/develop/doc/tutorial/climada_hazard_RiverFlood.ipynb

For each step undertaken in the paper we provide a demo tutorial under

https://github.com/ingajsa/flood_attribution_paper/tree/main/code/Demo/Demo_Scripts

The first tutorial how to generate damage for R, R+ and R- areas of Switzerland with CLIMADA.
The output of the other tutorials are examples for gaining the results presented in Fig.2-5 and Fig. SI1 and SI2
for the example of Latin America. Please note that results are only partly compareable to paper results,
as we are not allowed to publish observed damages. Outputs are small dataframes and plots for Latin America.

4 INSTRUCTIONS FOR USE

Afterwards, the demo tutorials on the present GitHub 
https://github.com/ingajsa/flood_attribution_paper/tree/main/code/Demo/Demo_Scripts
should be worked through in the following order:

1. demo_climada_damage_generation.ipynb 

2. demo_data_aggregation.ipynb 

3. demo_vulnerability_assessment.ipynb

4. demo_attribution_assessment.ipynb

5. demo_teleconnetions.ipynb

Only for the first tutorial CLIMADA needs to be installed and its conda environment needs to be activated.
Tutorials 2-5 can be run under any Python 3 environment and do not use CLIMADA. Alternatively,
one who is only iterested in the postprocessing can easily start with the 2nd Tutorial. To start with the
second tutorial only 

Please note that only dummies are provided for observational data, as we have no rights to publish the data_sets.

For the use of the demo data just start the jupyter-notebook 'demo_data_aggregation.ipynb' in DEMO_scripts and
follow the instructions. Only the input data set provided at /code/Demo/Demo_Data/demo_assembled_data_subregions.csv
is needed. Further input is generated when the instructions are followed.

------------------------------------------------------------------------------------------------------------
0 PREPROCESSING

The modeling of spatially explicit flood depth and fraction with CaMa-Flood and additional post-processing
can be accessed under 10.5281/zenodo.1241051 
The code provided there permits to generate the data on discharge, flood depth and flooded fraction provided
as input data for this study at https://files.isimip.org/cama-flood/results/isimip2a/

1 DAMAGE GENENERATION

The modeling process starts with the country based damage calculation with the impact modeling framework
CLIMADA available under:

https://github.com/CLIMADA-project/climada_python

Installation requirements and instructions are provided in the corresponding documentation.

In order to run the model, spatially explicit flooded fractions and flood depth provided by the ISIMIP2a
simulation round are required, these data are available under:

10.5281/zenodo.4446364

The script "schedule_runs.py"  in /code/scripts_reconstruction/run_climada/ a modeling run for each
climate forcing-GHM combination and calls the simulation script "schedule_sim.py". In "schedule_runs.py"
all climate forcing datasets and GHMs are defined.

The script calculates the flood damage for each country and every year and needs to be run for each
combination of climate forcing and GHM, see scripts in

/code/scripts_reconstruction/run_climada/

The nomenclature of the files in 10.5281/zenodo.4446364 is
described here:

https://github.com/ingajsa/flood_attribution_paper/blob/main/code/Demo/Demo_Scripts/demo_climada_damage_generation.ipynb

The output of "schedule_sim.py" are 46 .csv files containing damage-time series for each country between 1971-2010.

The further data needed to run the scripts is provided in the data folder or can be generated with the help
of the tutorial

https://github.com/ingajsa/flood_attribution_paper/blob/main/code/Demo/Demo_Scripts/demo_climada_damage_generation.ipynb

where links to the scripts needed for full reconstruction are provided.


2 POST-PROCESSING

The entire post-processing analysis is done once on regional level and on subregional level. 
Scripts ending with '...regions.py' are used to analyse entire regions, while scripts ending with
'...subregions.py' are for the analysis of subregions. Datasets derived from scripts with the Ending
'...regions.py' have to be used as an input for Scripts with the Ending '...regions.py', similarly 
datasets from scripts with the Ending'...subregions.py' have to be used as an input for Scripts with
the Ending '...subregions.py'

2.1 DATA AGGREGATION

In the first step, data is aggregated to regional/subregional level and across all model-runs, so damage time-series
aggregated to model-medians for each region/subregion are the output. Additionally, observed damages and country specific 
indicators are added and aggregated. For the aggregation all the output files of the 'schedule_runs.py' and the
'schedule_sim.py' script need to be accessed by both scripts: data_aggregation_regions.py and data_aggregation_subegions.py

2.2 VULNERABILITY ASSESSMENT

The aggregated files are then used for the vulnerability assessment in 'vulnerability_adjustment_regions.py' 
and 'vulnerability_adjustment_subregions.py' , further input is not necessary. The scripts each provide a MetaData and a
TimeSeries dataset which are then used for the attribution scripts. The MetaData contains information on explained variances 
and correlation.

2.3 ATTRIBUTION ASSESSMENT

The TimeSeries output is then used as an input for the scripts 'attributionRegions.py' and 'attributionSubregions.py'.
The Scripts again produce TimeSeries and MetaData which can than used to produce the Plots 2,3 and 4.
Both data sets serve as input for the detection of teleconnections. MetaData contains climate (H), exposure (E) and 
vulnerability (V) contributions as well as modeled (M) and observed trend (N) as well as significances.

2.4 DRIVERS FOR CLIMATE-INDUCED TRENDS

The two data sets generated during the attribution assessment are used as inputs for the scripts 'teleconnections_regions.py'
and 'teleconnections_subregions.py'. Climate Oscillation Indices need to be added and are available under:
Southern Oscillation Index as a predictor for ENSO (https://www.ncdc.noaa.gov/teleconnections/enso/enso-tech.php)
Monthly data for AMO, NAO and PDO were extracted from the NOAA/Climate Prediction Center
(https://www.psl.noaa.gov/data/climateindices/list/) they need to be centered and scaled prior to the analysis.


3 PLOTTING

The plot scripts are named according to their Figures in the papers. Which datasets are needed to produce
the plot is indicated in the script. 
