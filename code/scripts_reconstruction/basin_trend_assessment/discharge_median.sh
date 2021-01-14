#!/bin/bash
echo median_calculation

cdo enspctl,50 $all $discharge $files Median_All_1971_2001.nc 
cdo enspctl,50 $discharge $files $not $forced $by $watch Median_non-watch_1971_2010.nc 
cdo selyear,2002/2010  Median_non-watch_1971_2010.nc Median_non-watch_2002_2010.nc
cdo mergetime Median_All_1971_2001.nc  Median_NonWatch_2002_2010_24_2.nc Median_Merged_1971_2010.nc

echo end