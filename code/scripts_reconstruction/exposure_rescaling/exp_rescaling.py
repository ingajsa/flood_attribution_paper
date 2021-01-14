import numpy as np
import pandas as pd
from climada.entity.exposures.gdp_asset import GDP2Asset
from climada.util.constants import RIVER_FLOOD_REGIONS_CSV

gdp_pc = pd.read_csv('~/data/exposure_rescaling/' +
                     'Income-PPP2005_ISIMIP_merged_Maddison-pwt81_1850-2015_extended_WDI-1.csv')

pop = pd.read_csv('~/data/exposure_rescaling/' +
                  'Population_ISIMIP-pwt81_1850-2009_extended_WDI_filled-with-Hyde.csv')

# insert gridded gdp file here  https://doi.org/10.5880/pik.2017.007 'GCP_PPP-2005_1850-2100.nc'
# file needs to be interpolated to yearly data and upscaled to 2.5 arcmin

gdp_path = '~/data/downloads/gridded_gdp.nc'

# insert pwt data here https://www.rug.nl/ggdc/productivity/pwt/
pwd = pd.read_excel('~/data/downloads/pwt91.xlsx', sheet_name='Data')

gdp2asset = pd.read_excel('/home/insauer/projects/Attribution/Data/' +
                          'global-wealth-databook-2016_extract+ISO.xlsx')

years = np.arange(1950, 2012)
str_years = years.astype(str)
country_info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
isos = country_info['ISO'].tolist()

rs_dataDF = pd.DataFrame(columns=str_years)
rs_dataDF.insert(0, "ISO", isos)

ppco_dataDF = pd.DataFrame(columns=str_years)
ppco_dataDF.insert(0, "ISO", isos)


for iso in isos:
    conv_fac = gdp2asset.loc[gdp2asset['ISO'] == iso, 'all_wealth_ratio'].sum()

    for column, year in enumerate(years):
        print(str(year) + ' ' + iso)

        gdp_pc_cnt_yr = gdp_pc.loc[gdp_pc.iloc[:, 0] == iso, str(year)].sum()
        pop_cnt_yr = pop.loc[gdp_pc.iloc[:, 0] == iso, str(year)].sum()
        gdp_cnt_yr = gdp_pc_cnt_yr * pop_cnt_yr

        gdpa = GDP2Asset()
        gdpa.set_countries(countries=[iso], ref_year=year, path=gdp_path)

        gdp_jpn = gdpa['value'].sum()/conv_fac
        rs_dataDF.loc[rs_dataDF['ISO'] == iso, str(year)] = gdp_cnt_yr/gdp_jpn

        cgdpo = pwd.loc[(pwd['countrycode'] == iso) & (pwd['year']==year), 'rgdpna'].sum()
        cstock = pwd.loc[(pwd['countrycode'] == iso) & (pwd['year']==year), 'rnna'].sum()

        cgdpo_cstock = cstock/cgdpo

        if np.isnan(cgdpo_cstock):
            cgdpo_cstock = conv_fac

        if np.isinf(cgdpo_cstock):

            cgdpo_cstock = conv_fac

        pp_cgdpo = cgdpo_cstock/conv_fac

        if np.isnan(pp_cgdpo):
            pp_cgdpo = 1.

        if np.isinf(pp_cgdpo):

            pp_cgdpe = 1.

        ppco_dataDF.loc[ppco_dataDF['ISO']== iso, str(year)] = pp_cgdpo
        rs_dataDF.to_csv('~/data/exposure_rescaling/resc_ssp_transition_repro.csv')
        ppco_dataDF.to_csv('~/data/exposure_rescaling/totalwealth_capital_stock_rescaling.csv')

rs_dataDF.to_csv('~/data/exposure_rescaling/resc_ssp_transition_repro.csv')
ppco_dataDF.to_csv('~/data/exposure_rescaling/totalwealth_capital_stock_rescaling.csv')