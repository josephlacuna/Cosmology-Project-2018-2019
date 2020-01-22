import scipy as sp
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import numpy.polynomial.legendre as nppl
import scipy.constants as spc
import glob
import sys
from scipy.stats import linregress
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.table import Table
from djs_angle_match import djs_angle_match

'''
This code is from my summer internship at the University of Chicago, Kavli
Institute for Cosmological Physics. The work here is for calibrating data from
the PANSTARRS telescope, managing between multiple data releases and matching
objects from different telescope observations. The goal was to understand a
discrepancy in the filters which was made apparent when matching particular
calibration objects which we understand to be true.
'''

# Assigning Variables
nn = 3                                                    # n observation threshold
flux_min = 14.0                                           # Aperature flux min
flux_max = 19.0                                           # Aperature flux max
diff_min = -0.1                                           # Difference flux min
diff_max = 0.1                                            # Difference flux max
n_bins = 11                                               # number of bins
ErrBound = 0.05                                           # error bounds
color_bin_min = -1.0                                      # Color mag min
color_bin_max = 3.0                                       # Color mag max
minimum_dec = -30                                         # minimum declination

# Loading in Data

# Creating DF for Supercal Synthetic Mags
# supercal_obj = ['SF1615','SNAP-2','SNAP-1','WD1657','C26202','LDS749B']
syn_g = [16.996,16.447,15.495,16.224,16.676,14.573]
syn_r = [16.562,16.046,15.894,16.693,16.368,14.808]
syn_i = [16.385,15.904,16.202,17.073,16.263,15.039]
syn_z = [16.318,15.875,16.424,17.360,16.245,np.nan]
supercal_syn = pd.DataFrame(
    {'gSynMag':syn_g,
     'rSynMag':syn_r,
     'iSynMag':syn_i,
     'zSynMag':syn_z}
    )

# Loading Target Calibration Objects
targets = 'targets.xlsx'
target_DF = pd.read_excel(targets)
target_DF.dropna(inplace=True)
targ_mag = 'targ_mag.xlsx'
targ_mag_DF = pd.read_excel(targ_mag)

# filter by magnitude thresholds
lower_mag_thresh = 13
upper_mag_thresh = 20
targ_mag_DF = targ_mag_DF.loc[targ_mag_DF['V_mag'] >= lower_mag_thresh]
targ_mag_DF = targ_mag_DF.loc[targ_mag_DF['V_mag'] <= upper_mag_thresh]

# creating target DF
target_DF = pd.merge(target_DF,targ_mag_DF,how='inner',left_index=True,right_index=True)

# function for reformatting input .fits files into appropriate DF labels
def fits_to_df(path):
    dat = Table.read(path, format='fits')
    xdf = dat.to_pandas()
    xdf = xdf[['ra','dec','median(0)','median(1)','median(2)','median(3)',
    'mean(0)','mean(1)','mean(2)','mean(3)','err(0)','err(1)','err(2)','err(3)',
    'nmag_ok(0)','nmag_ok(1)','nmag_ok(2)','nmag_ok(3)',
    'median_ap(0)','median_ap(1)','median_ap(2)','median_ap(3)',
    'mean_ap(0)','mean_ap(1)','mean_ap(2)','mean_ap(3)',
    'err_ap(0)','err_ap(1)','err_ap(2)','err_ap(3)',
    'nmag_ap_ok(0)','nmag_ap_ok(1)','nmag_ap_ok(2)','nmag_ap_ok(3)']]
    xdf = xdf.rename(columns={'ra': 'ra','dec': 'dec',
    'median(0)': 'gMedianPSFMag','median(1)': 'rMedianPSFMag',
    'median(2)': 'iMedianPSFMag','median(3)': 'zMedianPSFMag',
    'mean(0)': 'gMeanPSFMag','mean(1)': 'rMeanPSFMag',
    'mean(2)': 'iMeanPSFMag','mean(3)': 'zMeanPSFMag',
    'err(0)': 'gPSFMagErr','err(1)': 'rPSFMagErr',
    'err(2)': 'iPSFMagErr','err(3)': 'zPSFMagErr',
    'nmag_ok(0)': 'ng_PSF','nmag_ok(1)': 'nr_PSF',
    'nmag_ok(2)': 'ni_PSF','nmag_ok(3)': 'nz_PSF',
    'median_ap(0)': 'gMedianApMag','median_ap(1)': 'rMedianApMag',
    'median_ap(2)': 'iMedianApMag','median_ap(3)': 'zMedianApMag',
    'mean_ap(0)': 'gMeanApMag','mean_ap(1)': 'rMeanApMag',
    'mean_ap(2)': 'iMeanApMag','mean_ap(3)': 'zMeanApMag',
    'err_ap(0)': 'gApMagErr','err_ap(1)': 'rApMagErr',
    'err_ap(2)': 'iApMagErr','err_ap(3)': 'zApMagErr',
    'nmag_ap_ok(0)': 'ng_Ap','nmag_ap_ok(1)': 'nr_Ap',
    'nmag_ap_ok(2)': 'ni_Ap','nmag_ap_ok(3)': 'nz_Ap'
    })
    return xdf


# Loading in PANSTARRS Data
# ucal_qy refers to a particular data release
panfiles = glob.glob("ucal_qy/*.csv")
DF = pd.read_csv(panfiles[0])
DF = DF.assign(file = panfiles[0])

for i in panfiles:
    if i == panfiles[0]:
        continue
    data = pd.read_csv(i)
    data = data.assign(file = i)
    DF = DF.append(data)

# reformats null data
DF = DF[(DF != -999).all(1)]

# ucal_qz Data [Eddie's Data]
# ucal_qy refers to a particular data release
qzfiles = glob.glob("ucal_qz/*.fits")
DF2 = fits_to_df(qzfiles[0])
DF2 = DF2.assign(file = qzfiles[0])

for i in qzfiles:
    if i == qzfiles[0]:
        continue
    qzdf = fits_to_df(i)
    qzdf = qzdf.assign(file=i)
    DF2 = DF2.append(qzdf)

# reformats null data
DF2 = DF2[(DF2!=0).all(1)]

# Wise Data
# wise refers to a particular data release
wisefiles = glob.glob("wise/*.fits")
DF3 = fits_to_df(wisefiles[0])
DF3 = DF3.assign(file = wisefiles[0])

for i in wisefiles:
    if i == wisefiles[0]:
        continue
    W_df = fits_to_df(i)
    W_df = W_df.assign(file = i)
    DF3 = DF3.append(W_df)

# reformats null data
DF3 = DF3[(DF3 != 0).all(1)]

allfiles = glob.glob("ucal_qz/*.fits")

# shortening/assigning variable names
ng = 'ng'
gPSF = 'gMeanPSFMag'
gAp = 'gMeanApMag'
gDiff = 'gMeanDiffMag'
nr = 'nr'
rPSF = 'rMeanPSFMag'
rAp = 'rMeanApMag'
rDiff = 'rMeanDiffMag'
ni = 'ni'
iPSF = 'iMeanPSFMag'
iAp = 'iMeanApMag'
iDiff = 'iMeanDiffMag'
nz = 'nz'
zPSF = 'zMeanPSFMag'
zAp = 'zMeanApMag'
zDiff = 'zMeanDiffMag'
gCDiff = 'g_colorDiff'
rCDiff = 'r_colorDiff'
iCDiff = 'i_colorDiff'
zCDiff = 'z_colorDiff'
color = 'color'
gApErr = 'gMeanApMagErr'
rApErr = 'rMeanApMagErr'
iApErr = 'iMeanApMagErr'
zApErr = 'zMeanApMagErr'
gMedianPSF = 'gMedianPSFMag'
rMedianPSF = 'rMedianPSFMag'
iMedianPSF = 'iMedianPSFMag'
zMedianPSF = 'zMedianPSFMag'
gMedianAp = 'gMedianApMag'
rMedianAp = 'rMedianApMag'
iMedianAp = 'iMedianApMag'
zMedianAp = 'zMedianApMag'
gMedianDiff = 'gMedianDiffMag'
rMedianDiff = 'rMedianDiffMag'
iMedianDiff = 'iMedianDiffMag'
zMedianDiff = 'zMedianDiffMag'
gDiffi = 'gMeanDiffMag_Interp'
rDiffi = 'rMeanDiffMag_Interp'
iDiffi = 'iMeanDiffMag_Interp'
zDiffi = 'zMeanDiffMag_Interp'
gDiffd = 'detrended_gDiff'
rDiffd = 'detrended_rDiff'
iDiffd = 'detrended_iDiff'
zDiffd = 'detrended_zDiff'
gMedianDiffi = 'gMedianDiffMag_Interp'
rMedianDiffi = 'rMedianDiffMag_Interp'
iMedianDiffi = 'iMedianDiffMag_Interp'
zMedianDiffi = 'zMedianDiffMag_Interp'
gMedianDiffd = 'detrended_gMedianDiff'
rMedianDiffd = 'detrended_rMedianDiff'
iMedianDiffd = 'detrended_iMedianDiff'
zMedianDiffd = 'detrended_zMedianDiff'

# cleaning DF3 to standardize variable names
DF3_col_list = [gMedianPSF,rMedianPSF,iMedianPSF,zMedianPSF,gPSF,rPSF,iPSF,zPSF,
gMedianAp,rMedianAp,iMedianAp,zMedianAp,gAp,rAp,iAp,zAp]
DF3_col_med_list = [gMedianPSF,rMedianPSF,iMedianPSF,zMedianPSF,gMedianAp,
rMedianAp,iMedianAp,zMedianAp]
DF3_col_err_list = ['gPSFMagErr','rPSFMagErr','iPSFMagErr','zPSFMagErr',
'gApMagErr','rApMagErr','iApMagErr','zApMagErr']

# finding Additive Constant
def flux_corr(x):
    flux_corr = -2.5 * np.log10(x)
    return flux_corr
def flux_calc(df,col_list,col_err_list,col_med_list):
    xdf = df.copy()
    for i in range(len(col_err_list)):
        flux_err_calc = xdf[col_err_list[i]] / xdf[col_med_list[i]]
        xdf[col_err_list[i]] = flux_err_calc
    for i in col_list:
        flux_calc = flux_corr(xdf[i])
        xdf[i] = flux_calc
    return xdf

# standardizing DFs with additive constant
DF2 = flux_calc(DF2, DF3_col_list,DF3_col_err_list,DF3_col_med_list)
DF3 = flux_calc(DF3, DF3_col_list,DF3_col_err_list,DF3_col_med_list)


# Filtering the Data
# DF = Public PANSTARRS Release; DF2 = Eddie's Data; DF3 = Wise Data
def filtering(ra,dec,PSF, Ap, n, Diff, ApErr, df,ID):
    xdf = df[[ra,dec,n, PSF, Ap, ApErr,ID]]                                # allocating a filter df
    # can add ra, dec here by putting in 'raMean', 'decMean'
    # xdf = xdf.loc[xdf[PSF] != -999.000]                   # filtering N/A values
    # xdf = xdf.loc[xdf[Ap] != -999.000]
    xdf = xdf.loc[xdf[ApErr] <= ErrBound]
    xdf = xdf.loc[xdf[Ap] >= flux_min]
    xdf = xdf.loc[xdf[Ap] <= flux_max]
    xdiff = xdf[PSF] - xdf[Ap]                            # calculating difference column
    xdf = xdf.assign(diff = xdiff)
    xdf = xdf.rename(columns={'diff': Diff})              # accurate renaming
    xdf = xdf.loc[xdf[Diff] >= diff_min]                  # filtering outlier differences
    xdf = xdf.loc[xdf[Diff] <= diff_max]
    xdf = xdf[xdf[n] > nn]                                # placing n observation threshold
    xdf = xdf.drop_duplicates()
    xdf.set_index([ra,dec,ID],inplace=True)
    return xdf

# PANSTARRS Public
gdf = filtering('raMean','decMean',gPSF, gAp, ng, gDiff, gApErr, DF,'objID')
rdf = filtering('raMean','decMean',rPSF, rAp, nr, rDiff, rApErr, DF,'objID')
idf = filtering('raMean','decMean',iPSF, iAp, ni, iDiff, iApErr, DF,'objID')
zdf = filtering('raMean','decMean',zPSF, zAp, nz, zDiff, zApErr, DF,'objID')
# Eddie's
E_gdf_median = filtering('ra','dec',gMedianPSF, gMedianAp, 'ng_Ap', gMedianDiff, 'gApMagErr', DF2,'file')
E_rdf_median = filtering('ra','dec',rMedianPSF, rMedianAp, 'nr_Ap', rMedianDiff, 'rApMagErr', DF2,'file')
E_idf_median = filtering('ra','dec',iMedianPSF, iMedianAp, 'ni_Ap', iMedianDiff, 'iApMagErr', DF2,'file')
E_zdf_median = filtering('ra','dec',zMedianPSF, zMedianAp, 'nz_Ap', zMedianDiff, 'zApMagErr', DF2,'file')
E_gdf_mean = filtering('ra','dec',gPSF, gAp, 'ng_Ap', gDiff, 'gApMagErr', DF2,'file')
E_rdf_mean = filtering('ra','dec',rPSF, rAp, 'nr_Ap', rDiff, 'rApMagErr', DF2,'file')
E_idf_mean = filtering('ra','dec',iPSF, iAp, 'ni_Ap', iDiff, 'iApMagErr', DF2,'file')
E_zdf_mean = filtering('ra','dec',zPSF, zAp, 'nz_Ap', zDiff, 'zApMagErr', DF2,'file')
# Wise
W_gdf = filtering('ra','dec',gMedianPSF, gMedianAp, 'ng_Ap', gMedianDiff, 'gApMagErr', DF3,'file')
W_rdf = filtering('ra','dec',rMedianPSF, rMedianAp, 'nr_Ap', rMedianDiff, 'rApMagErr', DF3,'file')
W_idf = filtering('ra','dec',iMedianPSF, iMedianAp, 'ni_Ap', iMedianDiff, 'iApMagErr', DF3,'file')
W_zdf = filtering('ra','dec',zMedianPSF, zMedianAp, 'nz_Ap', zMedianDiff, 'zApMagErr', DF3,'file')


# Merging the Data and Creating Color Mag Values
def merging(df, df_2, gAp, iAp):
    color_df = pd.merge(df,df_2,how='left',left_index=True,right_index=True)
    color_df = color_df.dropna()
    filter_colorDiff = color_df[gAp] - color_df[iAp]
    color_df = color_df.assign(color = filter_colorDiff)
    # color_df = color_df.loc[color_df[color] >= color_bin_min]
    # color_df = color_df.loc[color_df[color] <= color_bin_max]
    return color_df

# PANSTARRS Public
color_gdf = merging(gdf,idf,gAp,iAp)
color_rdf = pd.merge(color_gdf,rdf, how='left',left_index = True,right_index=True)
color_rdf = color_rdf.dropna()
color_idf = color_gdf.copy()
color_zdf = pd.merge(color_gdf,zdf, how='left',left_index = True,right_index=True)
color_zdf = color_zdf.dropna()
color_all = pd.merge(color_rdf,zdf, how='left',left_index=True,right_index=True)
color_all = color_all.dropna()
# Eddie's
color_E_gdf_median = merging(E_gdf_median, E_idf_median, gMedianAp, iMedianAp)
color_E_rdf_median = pd.merge(color_E_gdf_median, E_rdf_median, how='left',left_index = True,right_index=True)
color_E_rdf_median = color_E_rdf_median.dropna()
color_E_idf_median = color_E_gdf_median.copy()
color_E_zdf_median = pd.merge(color_E_gdf_median, E_zdf_median, how='left',left_index = True,right_index=True)
color_E_zdf_median = color_E_zdf_median.dropna()
color_E_gdf_mean = merging(E_gdf_mean, E_idf_mean, gAp, iAp)
color_E_rdf_mean = pd.merge(color_E_gdf_mean, E_rdf_mean, how='left',left_index = True,right_index=True)
color_E_rdf_mean = color_E_rdf_mean.dropna()
color_E_idf_mean = color_E_gdf_mean.copy()
color_E_zdf_mean = pd.merge(color_E_gdf_mean, E_zdf_mean, how='left',left_index = True,right_index=True)
color_E_zdf_mean = color_E_zdf_mean.dropna()
color_E_all_median = pd.merge(color_E_rdf_median,E_zdf_median,how='left',left_index=True,right_index=True)
color_E_all_median = color_E_all_median.dropna()
color_E_all_mean = pd.merge(color_E_rdf_mean,E_zdf_mean,how='left',left_index=True,right_index=True)
color_E_all_mean = color_E_all_mean.dropna()
# Wise
color_W_gdf = merging(W_gdf,W_idf,gMedianAp,iMedianAp)
color_W_rdf = pd.merge(color_W_gdf,W_rdf, how='left',left_index=True,right_index=True)
color_W_all = pd.merge(color_W_rdf,W_zdf, how='left',left_index=True,right_index=True)
color_W_all = color_W_all.dropna()
color_W_all_orig = color_W_all.copy()

# Tidying the target dataframe
target_DF.set_index(['ra','dec'],inplace=True)
target_DF = hdms_to_dd_modified(target_DF)
target_DF.reset_index(inplace=True)

# Restricting declination view
target_DF = target_DF.loc[target_DF['dd_dec'] >= minimum_dec]

# converting web RA and DEC and setting RA and DEC to columns
color_all_match = hdms_to_dd(color_all)
color_all_match.reset_index(inplace=True)
color_W_all.reset_index(inplace=True)

# Now that a lot of the data frames have been cleaned and standardized,
# we are able to now match objects between the DFs

# RA and DEC conversion functions between hdms units and dd units
def hdms_to_dd(df):
    index_list = df.index.tolist()
    ra_list = []
    dec_list = []
    for i in range(len(index_list)):
        ra = index_list[i][0]
        h_ra,m_ra,s_ra = ra.split(" ")
        dd_ra = round(float(15 * (float(h_ra)  + (float(m_ra) / 60) + (float(s_ra) / 3600))),6)

        dec = index_list[i][1]
        d_dec,m_dec,s_dec = dec.split(" ")
        if '-' in d_dec:
            dd_dec = round(float(float(d_dec) - (float(m_dec) / 60) - (float(s_dec)/3600)),6)
        else:
            dd_dec = round(float(float(d_dec) + (float(m_dec) / 60) + (float(s_dec)/3600)),6)

        ra_list.append(dd_ra)
        dec_list.append(dd_dec)
    df = df.assign(ra = ra_list)
    df = df.assign(dec = dec_list)
    return df

def hdms_to_dd_modified(df):
    index_list = df.index.tolist()
    ra_list = []
    dec_list = []
    for i in range(len(index_list)):
        # RA Parsing and Conversion
        ra = index_list[i][0]
        ra_parsed = ra.split(" ")
        if len(ra_parsed) == 3:
            dd_ra = round(float(15 * (float(ra_parsed[0])  + (float(ra_parsed[1]) / 60)
            + (float(ra_parsed[2]) / 3600))),6)
        if len(ra_parsed) == 2:
                dd_ra = round(float(15 * (float(ra_parsed[0])  + (float(ra_parsed[1]) / 60))),6)
        if len(ra_parsed) == 1:
                dd_ra = round(float(15 * float(ra_parsed[0])),6)
        # DEC Parsing and Conversion
        dec = index_list[i][1]
        dec_parsed = dec.split(" ")
        if '-' in dec_parsed[0]:
            if len(dec_parsed) == 3:
                dd_dec = round(float(float(dec_parsed[0]) - (float(dec_parsed[1]) / 60)
                - (float(dec_parsed[2])/3600)),6)
            if len(dec_parsed) == 2:
                dd_dec = round(float(float(dec_parsed[0]) - (float(dec_parsed[1]) / 60)),6)
            if len(dec_parsed) == 1:
                dd_dec = round(float(float(dec_parsed[0])),6)
        else:
            if len(dec_parsed) == 3:
                dd_dec = round(float(float(dec_parsed[0]) + (float(dec_parsed[1]) / 60)
                + (float(dec_parsed[2])/3600)),6)
            if len(dec_parsed) == 2:
                dd_dec = round(float(float(dec_parsed[0]) + (float(dec_parsed[1]) / 60)),6)
            if len(dec_parsed) == 1:
                dd_dec = round(float(float(dec_parsed[0])),6)
        ra_list.append(dd_ra)
        dec_list.append(dd_dec)
    df = df.assign(dd_ra = ra_list)
    df = df.assign(dd_dec = dec_list)
    return df

#  Matching by RA and DEC
#  matching web (DF) and edd (DF2) dataframes
ntot,mindx,mcount = djs_angle_match(color_all_match['ra'].values,
color_all_match['dec'].values, color_W_all['ra'].values, color_W_all['dec'].values, dtheta=0.005)
mindx = mindx[0]
xx = np.where((mindx >-1 ))

matching_web_DF = color_all_match.iloc[xx]
not_matched_web_DF = color_all_match.iloc[~color_all_match.index.isin(xx[0])]
matching_edd_DF = color_W_all.iloc[mindx[xx]]
not_matched_edd_DF = color_W_all.iloc[~color_W_all.index.isin(mindx[xx])]
matched_web_DF = matching_web_DF.reset_index()
matched_edd_DF = matching_edd_DF.reset_index()

# matched web and edd DF
matched_EW_DF = pd.merge(matched_edd_DF,matched_web_DF,how='left',left_index=True,right_index=True)
avg_color = (matched_EW_DF['color_x'] + matched_EW_DF['color_y']) / 2
matched_EW_DF = matched_EW_DF.assign(avg_color = avg_color)

# matching target objects to all three dataframes
target_ra = target_DF['dd_ra'].values
target_dec = target_DF['dd_dec'].values
match_dtheta = 0.005

# ucal_qy (PANSTARRS Public)
qy_ntot, qy_mindx, qy_mcount = djs_angle_match(color_all_match['ra'].values,
color_all_match['dec'].values,target_ra,target_dec,dtheta=match_dtheta)
qy_mindx = qy_mindx[0]
qy_xx = np.where((qy_mindx > -1))
matching_qy = color_all_match.iloc[qy_xx]
matching_qy_targets = target_DF.iloc[qy_mindx[qy_xx]]
matching_qy = matching_qy.assign(objID = matching_qy_targets['objID_x'].values)
matching_qy.set_index(['objID'],inplace=True)

# ucal_qz (Eddie's Data)
color_E_all_mean_matching = color_E_all_mean.reset_index()
qz_ntot, qz_mindx, qz_mcount = djs_angle_match(color_E_all_mean_matching['ra'].values,
color_E_all_mean_matching['dec'].values,target_ra,target_dec,dtheta=match_dtheta)
qz_mindx = qz_mindx[0]
qz_xx = np.where((qz_mindx > -1))
matching_qz = color_E_all_mean_matching.iloc[qz_xx]
matching_qz_targets = target_DF.iloc[qz_mindx[qz_xx]]
matching_qz = matching_qz.assign(objID = matching_qz_targets['objID_x'].values)
matching_qz.set_index(['objID'],inplace=True)

# wise public
pr_ntot, pr_mindx, pr_mcount = djs_angle_match(color_W_all['ra'].values,
color_W_all['dec'].values,target_ra,target_dec,dtheta=match_dtheta)
pr_mindx = pr_mindx[0]
pr_xx = np.where((pr_mindx > -1))
matching_pr = color_W_all.iloc[pr_xx]
matching_pr_targets = target_DF.iloc[pr_mindx[pr_xx]]
matching_pr = matching_pr.assign(objID = matching_pr_targets['objID_x'].values)
matching_pr.set_index(['objID'],inplace=True)
matching_targets = pd.merge(matching_qz,matching_pr,left_index=True,right_index=True).index.tolist()
matching_qz.reset_index(inplace=True)
matching_pr.reset_index(inplace=True)
matching_pr_qz = pd.merge(matching_pr,matching_qz,on='objID')

# getting WEB - ED for each filter
def mag_diff(df,x1,x2,xdif):
    x_diff = df[x1] - df[x2]
    df = df.assign(xdiff = x_diff)
    df = df.rename(columns={'xdiff': xdif})
    return df

matched_EW_DF = mag_diff(matched_EW_DF,gAp,gMedianAp,'gEW_Diff')
matched_EW_DF = mag_diff(matched_EW_DF,rAp,rMedianAp,'rEW_Diff')
matched_EW_DF = mag_diff(matched_EW_DF,iAp,iMedianAp,'iEW_Diff')
matched_EW_DF = mag_diff(matched_EW_DF,zAp,zMedianAp,'zEW_Diff')

def plot_matched(df,x_data,y_data1,y_data2,y_data3,y_data4,xlab,ylab,file_title,
ax1_title,ax2_title,ax3_title,ax4_title,size):
    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.scatter(x = df[x_data], y = df[y_data1],color='firebrick',s=size)
    ax2.scatter(x = df[x_data], y = df[y_data2],color='firebrick',s=size)
    ax3.scatter(x = df[x_data], y = df[y_data3],color='firebrick',s=size)
    ax4.scatter(x = df[x_data], y = df[y_data4],color='firebrick',s=size)
    ax3.set_xlabel(xlab)
    ax4.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax3.set_ylabel(ylab)
    ax1.set_title(ax1_title)
    ax2.set_title(ax2_title)
    ax3.set_title(ax3_title)
    ax4.set_title(ax4_title)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax1.axhline(y=0,color='k')
    ax2.axhline(y=0,color='k')
    ax3.axhline(y=0,color='k')
    ax4.axhline(y=0,color='k')
    ax1.set_ylim(0.075,0.0125)
    ax2.set_ylim(0.075,0.0125)
    ax3.set_ylim(0.075,0.0125)
    ax4.set_ylim(0.075,0.0125)
    fig.suptitle('{} vs. {} for filters G, R, I, Z'.format(ylab,xlab),
    x=0.5,y=0.95)
    fig.savefig(file_title, dpi = 600, bbox_inches = 'tight')
    plt.close()

#  Binning the Data #

def binning(xdf, x_val, nbins, lower_bin, upper_bin):
    bin_array = np.linspace(lower_bin, upper_bin, nbins)
    xdf['binned'] = pd.cut(xdf[x_val], bins = bin_array)
    # xdf = xdf.dropna()
    grouped_xdf = xdf.groupby('binned')
    binned_xdf = grouped_xdf.median()
    return binned_xdf

#  binning by Ap
binned_gdf = binning(color_gdf, gAp, n_bins, flux_min, flux_max)
binned_rdf = binning(color_rdf, rAp, n_bins, flux_min, flux_max)
binned_idf = binning(color_idf, iAp, n_bins, flux_min, flux_max)
binned_zdf = binning(color_zdf, zAp, n_bins, flux_min, flux_max)

binned_E_gdf_median = binning(color_E_gdf_median, gMedianAp, n_bins, flux_min, flux_max)
binned_E_rdf_median = binning(color_E_rdf_median, rMedianAp, n_bins, flux_min, flux_max)
binned_E_idf_median = binning(color_E_idf_median, iMedianAp, n_bins, flux_min, flux_max)
binned_E_zdf_median = binning(color_E_zdf_median, zMedianAp, n_bins, flux_min, flux_max)

binned_E_gdf_mean = binning(color_E_gdf_mean, gAp, n_bins, flux_min, flux_max)
binned_E_rdf_mean = binning(color_E_rdf_mean, rAp, n_bins, flux_min, flux_max)
binned_E_idf_mean = binning(color_E_idf_mean, iAp, n_bins, flux_min, flux_max)
binned_E_zdf_mean = binning(color_E_zdf_mean, zAp, n_bins, flux_min, flux_max)

# f_diff = flux_max - flux_min
# mid_bin_array = np.linspace(flux_min + (f_diff / ((n_bins-1)*2)), flux_max - (f_diff / ((n_bins-1)*2)), n_bins-1)

#  binning by color
binned_color_gdf = binning(color_gdf, color, n_bins, color_bin_min, color_bin_max)
binned_color_rdf = binning(color_rdf, color, n_bins, color_bin_min, color_bin_max)
binned_color_idf = binning(color_idf, color, n_bins, color_bin_min, color_bin_max)
binned_color_zdf = binning(color_zdf, color, n_bins, color_bin_min, color_bin_max)

binned_color_E_gdf_median = binning(color_E_gdf_median, color, n_bins, color_bin_min, color_bin_max)
binned_color_E_rdf_median = binning(color_E_rdf_median, color, n_bins, color_bin_min, color_bin_max)
binned_color_E_idf_median = binning(color_E_idf_median, color, n_bins, color_bin_min, color_bin_max)
binned_color_E_zdf_median = binning(color_E_zdf_median, color, n_bins, color_bin_min, color_bin_max)

binned_color_E_gdf_mean = binning(color_E_gdf_mean, color, n_bins, color_bin_min, color_bin_max)
binned_color_E_rdf_mean = binning(color_E_rdf_mean, color, n_bins, color_bin_min, color_bin_max)
binned_color_E_zdf_mean = binning(color_E_zdf_mean, color, n_bins, color_bin_min, color_bin_max)
binned_color_E_idf_mean = binning(color_E_idf_mean, color, n_bins, color_bin_min, color_bin_max)

binned_matched_EW_DF = binning(matched_EW_DF, 'avg_color', n_bins, color_bin_min, color_bin_max)
binned_color_W_all = binning(color_W_all_orig, 'color', n_bins, color_bin_min, color_bin_max)

#  Sigma Clipping Process
sig = 2
itr = 1
leg_num = 1

def regression(DF):
    x=np.arange(DF.size)
    m,b,r,p,e = linregress(x,DF)
    yval = m * x[-1] + b
    return yval

def clipping(xdf1, Diff, x_var):
    # xdf1['Std'] = xdf1[Diff].rolling(rolls).std()
    # xdf1['Median'] = xdf1[Diff].rolling(rolls).median()
    # print(xdf1)
    xdf = xdf1.copy()
    '''
    for i in range(0,iters):
        upper = xdf['Median'] + sigma_upper * xdf['Std']
        lower = xdf['Median'] - sigma_lower * xdf['Std']
        mask = xdf[Diff] < upper
        mask = xdf[Diff] > lower
        xdf = xdf[mask]
        xStd = xdf[Diff].rolling(rolls).std()
        xdf = xdf.assign(Std = xStd)
        xMedian = xdf[Diff].rolling(rolls).agg(regression)
        xdf = xdf.assign(Median = xMedian)
    '''
    mask = sigma_clip(xdf[Diff], sigma = sig, iters = itr)
    xdf = xdf.assign(maskDiff = mask)
    coefs = nppl.legfit(xdf[x_var],xdf['maskDiff'],leg_num)
    fit = nppl.legval(xdf[x_var],coefs)
    return xdf, coefs, fit

# df2 denotes the masked dataframe
binned_gdf2, gcoefs, gfit = clipping(binned_gdf, gDiff, gAp)
binned_rdf2, rcoefs, rfit = clipping(binned_rdf, rDiff, rAp)
binned_idf2, icoefs, ifit = clipping(binned_idf, iDiff, iAp)
binned_zdf2, zcoefs, zfit = clipping(binned_zdf, zDiff, zAp)

binned_color_gdf2, color_gcoefs, color_gfit = clipping(binned_color_gdf, gDiff, color)
binned_color_rdf2, color_rcoefs, color_rfit = clipping(binned_color_rdf, rDiff, color)
binned_color_idf2, color_icoefs, color_ifit = clipping(binned_color_idf, iDiff, color)
binned_color_zdf2, color_zcoefs, color_zfit = clipping(binned_color_zdf, zDiff, color)

binned_E_gdf2_median, E_gcoefs_median, E_gfit_median = clipping(binned_E_gdf_median, gMedianDiff, gMedianAp)
binned_E_rdf2_median, E_rcoefs_median, E_rfit_median = clipping(binned_E_rdf_median, rMedianDiff, rMedianAp)
binned_E_idf2_median, E_icoefs_median, E_ifit_median = clipping(binned_E_idf_median, iMedianDiff, iMedianAp)
binned_E_zdf2_median, E_zcoefs_median, E_zfit_median = clipping(binned_E_zdf_median, zMedianDiff, zMedianAp)

binned_color_E_gdf2_median, color_E_gcoefs_median, color_E_gfit_median = clipping(binned_color_E_gdf_median,
gMedianDiff, color)
binned_color_E_rdf2_median, color_E_rcoefs_median, color_E_rfit_median = clipping(binned_color_E_rdf_median,
rMedianDiff, color)
binned_color_E_idf2_median, color_E_icoefs_median, color_E_ifit_median = clipping(binned_color_E_idf_median,
iMedianDiff, color)
binned_color_E_zdf2_median, color_E_zcoefs_median, color_E_zfit_median = clipping(binned_color_E_zdf_median,
zMedianDiff, color)

binned_E_gdf2_mean, E_gcoefs_mean, E_gfit_mean = clipping(binned_E_gdf_mean, gDiff, gAp)
binned_E_rdf2_mean, E_rcoefs_mean, E_rfit_mean = clipping(binned_E_rdf_mean, rDiff, rAp)
binned_E_idf2_mean, E_icoefs_mean, E_ifit_mean = clipping(binned_E_idf_mean, iDiff, iAp)
binned_E_zdf2_mean, E_zcoefs_mean, E_zfit_mean = clipping(binned_E_zdf_mean, zDiff, zAp)

binned_color_E_gdf2_mean, color_E_gcoefs_mean, color_E_gfit_mean = clipping(binned_color_E_gdf_mean,
gDiff, color)
binned_color_E_rdf2_mean, color_E_rcoefs_mean, color_E_rfit_mean = clipping(binned_color_E_rdf_mean,
rDiff, color)
binned_color_E_idf2_mean, color_E_icoefs_mean, color_E_ifit_mean = clipping(binned_color_E_idf_mean,
iDiff, color)
binned_color_E_zdf2_mean, color_E_zcoefs_mean, color_E_zfit_mean = clipping(binned_color_E_zdf_mean,
zDiff, color)

binned_color_W_gdf2, color_W_gcoefs, color_W_gfit = clipping(binned_color_W_all,
gMedianDiff, color)
binned_color_W_rdf2, color_W_rcoefs, color_W_rfit = clipping(binned_color_W_all,
rMedianDiff, color)
binned_color_W_idf2, color_W_icoefs, color_W_ifit = clipping(binned_color_W_all,
iMedianDiff, color)
binned_color_W_zdf2, color_W_zcoefs, color_W_zfit = clipping(binned_color_W_all,
zMedianDiff, color)

# Getting Index
def fit_index(fit):
    fit_bin_array = []
    for i in range(0,len(fit)):
        mid_val = (fit.index.values[i].right - fit.index.values[i].left) / 2
        mid = fit.index.values[i].left + mid_val
        fit_bin_array.append(mid)
    return fit_bin_array

#  Detrending Color Dependency #

def color_detrend(color_DF, binned_color_DF2, Diff, filterDiffi, filterDiffd):
    filterDiff_interp = np.interp(color_DF[color], binned_color_DF2[color], binned_color_DF2[Diff])
    color_DF = color_DF.assign(filterMeanDiffMag_Interp = filterDiff_interp)
    detr_filter_Diff = color_DF[Diff] - (color_DF['filterMeanDiffMag_Interp'] * 1 )
    color_DF = color_DF.assign(detrended_filterDiff = detr_filter_Diff)
    color_DF = color_DF.rename(columns={'filterMeanDiffMag_Interp': filterDiffi,
    'detrended_filterDiff': filterDiffd})
    return color_DF

detr_color_gdf = color_detrend(color_gdf, binned_color_gdf2, gDiff, gDiffi, gDiffd)
detr_color_rdf = color_detrend(color_rdf, binned_color_rdf2, rDiff, rDiffi, rDiffd)
detr_color_idf = color_detrend(color_idf, binned_color_idf2, iDiff, iDiffi, iDiffd)
detr_color_zdf = color_detrend(color_zdf, binned_color_zdf2, zDiff, zDiffi, zDiffd)

bin_detrend_color_gdf = binning(detr_color_gdf, gAp, n_bins, flux_min, flux_max)
bin_detrend_color_rdf = binning(detr_color_rdf, rAp, n_bins, flux_min, flux_max)
bin_detrend_color_idf = binning(detr_color_idf, iAp, n_bins, flux_min, flux_max)
bin_detrend_color_zdf = binning(detr_color_zdf, zAp, n_bins, flux_min, flux_max)

detr_color_E_gdf_median = color_detrend(color_E_gdf_median, binned_color_E_gdf2_median,
gMedianDiff,gMedianDiffi,gMedianDiffd)
detr_color_E_rdf_median = color_detrend(color_E_rdf_median, binned_color_E_rdf2_median,
rMedianDiff,rMedianDiffi,rMedianDiffd)
detr_color_E_idf_median = color_detrend(color_E_idf_median, binned_color_E_idf2_median,
iMedianDiff,iMedianDiffi,iMedianDiffd)
detr_color_E_zdf_median = color_detrend(color_E_zdf_median, binned_color_E_zdf2_median,
zMedianDiff,zMedianDiffi,zMedianDiffd)

bin_detrend_color_E_gdf_median = binning(detr_color_E_gdf_median,gMedianAp,n_bins,flux_min,flux_max)
bin_detrend_color_E_rdf_median = binning(detr_color_E_rdf_median,rMedianAp,n_bins,flux_min,flux_max)
bin_detrend_color_E_idf_median = binning(detr_color_E_idf_median,iMedianAp,n_bins,flux_min,flux_max)
bin_detrend_color_E_zdf_median = binning(detr_color_E_zdf_median,zMedianAp,n_bins,flux_min,flux_max)

detr_color_E_gdf_mean = color_detrend(color_E_gdf_mean, binned_color_E_gdf2_mean,
gDiff,gDiffi,gDiffd)
detr_color_E_rdf_mean = color_detrend(color_E_rdf_mean, binned_color_E_rdf2_mean,
rDiff,rDiffi,rDiffd)
detr_color_E_idf_mean = color_detrend(color_E_idf_mean, binned_color_E_idf2_mean,
iDiff,iDiffi,iDiffd)
detr_color_E_zdf_mean = color_detrend(color_E_zdf_mean, binned_color_E_zdf2_mean,
zDiff,zDiffi,zDiffd)

bin_detrend_color_E_gdf_mean = binning(detr_color_E_gdf_mean,gAp,n_bins,flux_min,flux_max)
bin_detrend_color_E_rdf_mean = binning(detr_color_E_rdf_mean,rAp,n_bins,flux_min,flux_max)
bin_detrend_color_E_idf_mean = binning(detr_color_E_idf_mean,iAp,n_bins,flux_min,flux_max)
bin_detrend_color_E_zdf_mean = binning(detr_color_E_zdf_mean,zAp,n_bins,flux_min,flux_max)

#  PLOTTING #

dim_x = 6                               # x-dimension of plot resolution
dim_y = 4.5                             # y-dimension of plot resolution
lim_x = flux_min,flux_max               # limit of x-axis
lim_y = diff_min, diff_max              # limit of y-axis
dot_size = 10                           # size of scatter points
color_lim_x = color_bin_min, color_bin_max
g_xlabel = r'$g_{Ap}$'
g_ylabel = r'$g_{PSF} - g_{Ap}$'
r_xlabel = r'$r_{Ap}$'
r_ylabel = r'$r_{PSF} - r_{Ap}$'
i_xlabel = r'$i_{Ap}$'
i_ylabel = r'$i_{PSF} - i_{Ap}$'
z_xlabel = r'$z_{Ap}$'
z_ylabel = r'$z_{PSF} - z_{Ap}$'
color_label = r'$g_{Ap} - i_{Ap}$'


#  MATCHED WEB AND ED COLOR PLOT #

# corrections given by supercal

binned_matched_EW_DF['gEW_Diff'] -= 0.020
binned_matched_EW_DF['rEW_Diff'] -= 0.033
binned_matched_EW_DF['iEW_Diff'] -= 0.024
binned_matched_EW_DF['zEW_Diff'] -= 0.028

#  CALCULATING DETREND ON PUBLIC RELEASE #

g_SynFitColor = (matching_qy_supercal[color] * color_gcoefs[1]) + color_gcoefs[0]
r_SynFitColor = (matching_qy_supercal[color] * color_rcoefs[1]) + color_rcoefs[0]
i_SynFitColor = (matching_qy_supercal[color] * color_icoefs[1]) + color_icoefs[0]
z_SynFitColor = (matching_qy_supercal[color] * color_zcoefs[1]) + color_zcoefs[0]
matching_qy_supercal = matching_qy_supercal.assign(g_SynDiffFit = g_SynFitColor)
matching_qy_supercal = matching_qy_supercal.assign(r_SynDiffFit = r_SynFitColor)
matching_qy_supercal = matching_qy_supercal.assign(i_SynDiffFit = i_SynFitColor)
matching_qy_supercal = matching_qy_supercal.assign(z_SynDiffFit = z_SynFitColor)
# adding detrended difference
g_Detr_PSFMag = matching_qy_supercal[gPSF] - g_SynDiffFit
r_Detr_PSFMag = matching_qy_supercal[rPSF] - r_SynDiffFit
i_Detr_PSFMag = matching_qy_supercal[iPSF] - i_SynDiffFit
z_Detr_PSFMag = matching_qy_supercal[zPSF] - z_SynDiffFit
matching_qy_supercal = matching_qy_supercal.assign(g_Detr_PSFMag = g_Detr_PSFMag)
matching_qy_supercal = matching_qy_supercal.assign(r_Detr_PSFMag = r_Detr_PSFMag)
matching_qy_supercal = matching_qy_supercal.assign(i_Detr_PSFMag = i_Detr_PSFMag)
matching_qy_supercal = matching_qy_supercal.assign(z_Detr_PSFMag = z_Detr_PSFMag)
g_Detr_PSF_SynDiffMag = matching_qy_supercal['g_Detr_PSFMag'].values - supercal_syn['gSynMag'].values
r_Detr_PSF_SynDiffMag = matching_qy_supercal['r_Detr_PSFMag'].values - supercal_syn['rSynMag'].values
i_Detr_PSF_SynDiffMag = matching_qy_supercal['i_Detr_PSFMag'].values - supercal_syn['iSynMag'].values
z_Detr_PSF_SynDiffMag = matching_qy_supercal['z_Detr_PSFMag'].values - supercal_syn['zSynMag'].values
matching_qy_supercal = matching_qy_supercal.assign(g_Detr_PSF_SynDiffMag = g_Detr_PSF_SynDiffMag)
matching_qy_supercal = matching_qy_supercal.assign(r_Detr_PSF_SynDiffMag = r_Detr_PSF_SynDiffMag)
matching_qy_supercal = matching_qy_supercal.assign(i_Detr_PSF_SynDiffMag = i_Detr_PSF_SynDiffMag)
matching_qy_supercal = matching_qy_supercal.assign(z_Detr_PSF_SynDiffMag = z_Detr_PSF_SynDiffMag)

# detrended PSF - Syn
g_Detr_SynDiffMag = g_Detr_PSFMag.values - supercal_syn['gSynMag'].values
r_Detr_SynDiffMag = r_Detr_PSFMag.values - supercal_syn['rSynMag'].values
i_Detr_SynDiffMag = i_Detr_PSFMag.values - supercal_syn['iSynMag'].values
z_Detr_SynDiffMag = z_Detr_PSFMag.values - supercal_syn['zSynMag'].values
matching_qy_supercal = matching_qy_supercal.assign(g_Detr_SynDiffMag = g_Detr_SynDiffMag)
matching_qy_supercal = matching_qy_supercal.assign(r_Detr_SynDiffMag = r_Detr_SynDiffMag)
matching_qy_supercal = matching_qy_supercal.assign(i_Detr_SynDiffMag = i_Detr_SynDiffMag)
matching_qy_supercal = matching_qy_supercal.assign(z_Detr_SynDiffMag = z_Detr_SynDiffMag)

#  SINGLE FILTER PLOTS #

def plot_single_uncorrected(x_data,y_data,filterfit,filtercoefs,color_filterdf,xlab,ylab,title):
    fig = plt.figure(figsize=(dim_x,dim_y))
    ax = fig.add_subplot(111)
    # ax.scatter(x = gAp, y = gDiff, s = dot_size, color = 'black')
    ax.scatter(x = x_data, y = y_data, s = 20, color = 'firebrick')
    ax.axhline(y=0,color='k')
    # ax.scatter(x = mid_bin_array, y = binned_gdf[gDiff], s = dot_size, color = 'firebrick')
    # ax.plot(g_bin_array, filterfit.values, 'k-')
    # ax.text(0.6,0.2,"y = {}x + {} \n {} Observations".format(np.round(float(filtercoefs[1]),5), np.round(float(filtercoefs[0]),5), color_filterdf.shape[0]), transform=ax.transAxes)
    # xlab = r'$g_{Ap}$'
    # ylab = r'$g_{PSF} - g_{Ap}$'
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title('{} vs. {}'.format(ylab,xlab))
    ax.grid(True)
    ax.set_xlim(-1.0,1.0)
    ax.set_ylim(-0.015,0.015)
    ax.text(0.6,0.013,"Detrended")
    fig.savefig(title, dpi = 600, bbox_inches = 'tight')
    plt.close()

g_ylabel_syn_P = r'$g_{PSF} - g_{Syn}$'
r_ylabel_syn_P = r'$r_{PSF} - r_{Syn}$'
i_ylabel_syn_P = r'$i_{PSF} - i_{Syn}$'
z_ylabel_syn_P = r'$z_{PSF} - z_{Syn}$'
g_ylabel_syn_A = r'$g_{Ap} - g_{Syn}$'
r_ylabel_syn_A = r'$r_{Ap} - r_{Syn}$'
i_ylabel_syn_A = r'$i_{Ap} - i_{Syn}$'
z_ylabel_syn_A = r'$z_{Ap} - z_{Syn}$'

# not detrended
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['gPSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, g_ylabel_syn_P,"PR_Syn_g_PSF_1.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['rPSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, r_ylabel_syn_P,"PR_Syn_r_PSF_1.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['iPSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, i_ylabel_syn_P,"PR_Syn_i_PSF_1.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['zPSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, z_ylabel_syn_P,"PR_Syn_z_PSF_1.jpg")
'''
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['gAp_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, g_ylabel_syn_A,"PR_Syn_g_Ap.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['rAp_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, r_ylabel_syn_A,"PR_Syn_r_Ap.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['iAp_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, i_ylabel_syn_A,"PR_Syn_i_Ap.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['zAp_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, z_ylabel_syn_A,"PR_Syn_z_Ap.jpg")
'''
# detrended
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['g_Detr_PSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, g_ylabel_syn_P,"PR_Syn_g_PSF_detr_1.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['r_Detr_PSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, r_ylabel_syn_P,"PR_Syn_r_PSF_detr_1.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['i_Detr_PSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, i_ylabel_syn_P,"PR_Syn_i_PSF_detr_1.jpg")
plot_single_uncorrected(matching_qy_supercal[color], matching_qy_supercal['z_Detr_PSF_SynDiffMag'],
E_gfit_mean, E_gcoefs_mean,color_E_gdf_mean, color_label, z_ylabel_syn_P,"PR_Syn_z_PSF_detr_1.jpg")

#  COLOR PLOTS #

def plot_single_color(x_data,y_data,filterfit1,filtercoefs1,color_filterdf1,ylab,title,
x2_data,y2_data,x3_data,y3_data,filterfit2,filtercoefs2,filterfit3,filtercoefs3,
color_filterdf2,color_filterdf3,x_target_data,y_target_data,x_target2_data,y_target2_data,
matched_target_data,x_matched_target_data,y_matched_target_data,x2_matched_target_data,
y2_matched_target_data,x_target3_data,y_target3_data):
    fig = plt.figure(figsize=(dim_x,dim_y))
    ax = fig.add_subplot(111)
    ax.scatter(x = x_data, y = y_data, s = dot_size, color = 'firebrick',label='Public Release')
    ax.scatter(x = x2_data, y = y2_data, s = dot_size, color = 'k',label= 'ucal_qz')
    ax.scatter(x = x3_data, y = y3_data, s = dot_size, color = 'midnightblue', label = 'ucal_qy')
    ax.plot(x_data, filterfit1.values, c = 'firebrick', ls = '-')
    ax.plot(x2_data, filterfit2.values, c = 'k', ls = '--')
    ax.plot(x3_data, filterfit3.values, c = 'midnightblue', ls = '-.')
    ax.text(0.1,0.3,"y = {}x + {} \n {} Observations".format(np.round(float(filtercoefs1[1]),5),
    np.round(float(filtercoefs1[0]),5), color_filterdf1.shape[0]), transform=ax.transAxes,
    color = 'firebrick', size = 'smaller')
    ax.text(0.1,0.2,"y = {}x + {} \n {} Observations".format(np.round(float(filtercoefs2[1]),5),
    np.round(float(filtercoefs2[0]),5), color_filterdf2.shape[0]), transform=ax.transAxes,
    color = 'k', size = 'smaller')
    ax.text(0.1,0.1,"y = {}x + {} \n {} Observations".format(np.round(float(filtercoefs3[1]),5),
    np.round(float(filtercoefs3[0]),5), color_filterdf3.shape[0]), transform=ax.transAxes,
    color = 'midnightblue', size = 'smaller')
    # calspec objects

    ax.scatter(x = x_target_data, y = y_target_data, s = dot_size+20, color = 'k',
    edgecolors ='silver',label='calspec_qz')
    ax.scatter(x = x_target2_data, y = y_target2_data, color = 'midnightblue',
    edgecolors = 'gold',label='calspec_qy',s = dot_size+20)
    ax.scatter(x = x_target3_data, y = y_target3_data, color = 'firebrick',
    edgecolors = 'indigo',label='calspec_PR',s = dot_size+20)

    pr_cal = [44749,52324,46012,23166,23572,58715]
    x4 = x_target3_data.loc[pr_cal]
    y4 = y_target3_data.loc[pr_cal]
    ax.scatter(x = x4, y = y4, color='firebrick',edgecolors='indigo',
    label='calspec_PR_Super',s=60,marker='*')
    '''
    for i in range(matched_target_data.shape[0]):
        x = [x_matched_target_data[i],x2_matched_target_data[i]]
        y = [y_matched_target_data[i],y2_matched_target_data[i]]
        ax.plot(x,y,'-',color='k',linewidth=0.5)
    '''
    xlab = r'$g_{Ap} - i_{Ap}$'
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title('{} vs. {}'.format(ylab,xlab))
    ax.grid(True)
    ax.set_xlim(color_lim_x)
    ax.set_ylim(-0.025,0.025)
    ax.legend(bbox_to_anchor=(1.0, 1.0),bbox_transform=plt.gcf().transFigure)
    fig.savefig(title, dpi = 600, bbox_inches = 'tight')
    plt.close()

version = "supercal"

plot_single_color(binned_color_gdf[color],binned_color_gdf[gDiff],color_gfit,
color_gcoefs,color_gdf,g_ylabel,"color_fig_g_qz_{}.jpg".format(version),binned_color_E_gdf_mean[color],
binned_color_E_gdf_mean[gDiff],binned_color_W_all[color],binned_color_W_all[gMedianDiff],
color_E_gfit_mean,color_E_gcoefs_mean,color_W_gfit,color_W_gcoefs,color_E_gdf_mean,
color_W_all,matching_qz[color],matching_qz[gDiff],matching_pr[color],matching_pr[gMedianDiff],
matching_pr_qz,matching_pr_qz['color_y'],matching_pr_qz[gDiff],matching_pr_qz['color_x'],
matching_pr_qz[gMedianDiff],matching_qy[color],matching_qy[gDiff])

plot_single_color(binned_color_rdf[color],binned_color_rdf[rDiff],color_rfit,
color_rcoefs,color_rdf,r_ylabel,"color_fig_r_qz_{}.jpg".format(version),binned_color_E_rdf_mean[color],
binned_color_E_rdf_mean[rDiff],binned_color_W_all[color],binned_color_W_all[rMedianDiff],
color_E_rfit_mean,color_E_rcoefs_mean,color_W_rfit,color_W_rcoefs,color_E_rdf_mean,
color_W_all,matching_qz[color],matching_qz[rDiff],matching_pr[color],matching_pr[rMedianDiff],
matching_pr_qz,matching_pr_qz['color_y'],matching_pr_qz[rDiff],matching_pr_qz['color_x'],
matching_pr_qz[rMedianDiff],matching_qy[color],matching_qy[rDiff])

plot_single_color(binned_color_idf[color],binned_color_idf[iDiff],color_ifit,
color_icoefs,color_idf,i_ylabel,"color_fig_i_qz_{}.jpg".format(version),binned_color_E_idf_mean[color],
binned_color_E_idf_mean[iDiff],binned_color_W_all[color],binned_color_W_all[iMedianDiff],
color_E_ifit_mean,color_E_icoefs_mean,color_W_ifit,color_W_icoefs,color_E_idf_mean,
color_W_all,matching_qz[color],matching_qz[iDiff],matching_pr[color],matching_pr[iMedianDiff],
matching_pr_qz,matching_pr_qz['color_y'],matching_pr_qz[iDiff],matching_pr_qz['color_x'],
matching_pr_qz[iMedianDiff],matching_qy[color],matching_qy[iDiff])

plot_single_color(binned_color_zdf[color],binned_color_zdf[zDiff],color_zfit,
color_zcoefs,color_zdf,z_ylabel,"color_fig_z_qz_{}.jpg".format(version),binned_color_E_zdf_mean[color],
binned_color_E_zdf_mean[zDiff],binned_color_W_all[color],binned_color_W_all[zMedianDiff],
color_E_zfit_mean,color_E_zcoefs_mean,color_W_zfit,color_W_zcoefs,color_E_zdf_mean,
color_W_all,matching_qz[color],matching_qz[zDiff],matching_pr[color],matching_pr[zMedianDiff],
matching_pr_qz,matching_pr_qz['color_y'],matching_pr_qz[zDiff],matching_pr_qz['color_x'],
matching_pr_qz[zMedianDiff],matching_qy[color],matching_qy[zDiff])
