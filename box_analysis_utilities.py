#!/usr/bin/env python
# coding: utf-8
# box_analysis_utilities.py
# Contact: sayrejay@gmail.com
import os, sys
import time, datetime

import regionmask
import geopandas as gpd
import shapely
import xarray as xr
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import cascaded_union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packaging import version
from functools import partial
import multiprocessing as mp
import itertools

import imageio

### Params
eng_or_esp = "eng" ###eng for english plots, else for spanish plots
asset = 'analytic_sr'

base_dir = "~/"
remote_sen_input_dir = os.path.join(base_dir, "Remote Sensing", "Input/")
remote_sen_output_dir = os.path.join(base_dir, "Remote Sensing", "Output/")
image_dir = os.path.join(base_dir, "Remote Sensing", "Images/")

data_dir = os.path.join(base_dir, "data")
intermediate_dir = os.path.join(base_dir, "Intermediates")

def clean_up_downloaded_tif(image_ids, chosenmun, ciclo, subset_df, asset_type='analytic_sr',
                            analysis_type='muns',state_dict={}):
    ### clean_up_downloaded_tif -- given that each box may have a certain number of images
    ### that cover (some part of) it, takes each image, subsets it down to the extent of the
    ### box/AOI (using the return_mask fxn above), and then averages those images together,
    ### returning the averaged matrix
    if analysis_type == "muns":
        if mex_or_mmr == "mmr":
            chosen_state = "MMR"+str(chosenmun).replace(str(chosenmun)[-3:],"")
        else:
            chosen_state = str(chosenmun).replace(str(chosenmun)[-3:],"")
    else:
        chosen_state = str(state_dict[chosenmun])

    da_full = xr.DataArray(data=0)
    crs_store = []
    for i, img_id in enumerate(image_ids):
        udm2_fl = os.path.join(image_dir,chosen_state,ciclo,"udm2",img_id+"_udm2.tif")
        udm_fl = os.path.join(image_dir,chosen_state,ciclo,"udm",img_id+"_udm.tif")
        if asset_type == "analytic":
            img_fl = os.path.join(image_dir,chosen_state,ciclo,"analytic",img_id+"_a.tif")
        else:
            img_fl = os.path.join(image_dir,chosen_state,ciclo,"analytic_sr",img_id+"_sr.tif")
        if os.path.isfile(img_fl):
            try:
                da = xr.open_rasterio(img_fl)
                img_read_success = True
            except:
                print("Encountered file handling error for dir:", chosen_state, "cycle:", ciclo, "Img:", img_id)
                image_ids.remove(img_id)
                img_read_success = False

            if img_read_success:
                crs_store.append(str(da.crs))

                if subset_df.crs != da.crs:
                    subset_poly_img = subset_df.to_crs(da.crs)
                else:
                    subset_poly_img = subset_df

                try:
                    da = return_mask(da, subset_poly_img, chosenmun)
                    img_subset_success = True
                except:
                    image_ids.remove(img_id)
                    img_subset_success = False

                if os.path.isfile(udm2_fl):
                    udm_da = xr.open_rasterio(udm2_fl)[0,:,:] ### select clear layer
                    # clear_da = udm_da[0,:,:] ### Clear values represents w/o cloud, haze, shadow and/or snow (0: not clear, 1: clear)
                    # cloud_da = udm_da[5,:,:] ### Cloud map (0: no cloud, 1: cloud)
                    da = da.where(udm_da == 1)
                    # da = da.where(clear_da == 1) ### Make sure image is clear of clouds, haze, shadows, and snow
                    # da = da.where(cloud_da == 0) ### Only ensure image free of clouds
                elif os.path.isfile(udm_fl):
                    udm_da = xr.open_rasterio(udm_fl)[0,:,:]
                    da = da.where(udm_da == 0)

                da = da.where(da != 0)
                da = da/10000

                if img_subset_success:
                    if i == 0:
                        da_full = da
                    else:
                        da_full = xr.concat([da_full, da], "sat_images")
        else:
            print("Encountered file handling error for dir:", chosen_state, "cycle:", ciclo, "Img:", img_id)
            image_ids.remove(img_id)
    if len(image_ids) > 1:
        if 'sat_images' in da_full.dims:
            da_full = da_full.mean(dim='sat_images')
    return da_full, list(set(crs_store))

def return_mask(dataarray,geo_df,muncode=None,geo_col='muncode',mask_col='mask'):
    ### return_mask -- takes a dataarray (this is the xarray matrix corresponding
    ### to a satellite image), and masks it down to the area of interest, or AOI,
    ### given by the shapely object in geo_df, returning that dataarray. If muncode
    ### is not None, it subsets the dataarray down to that one municipality in geo_df
    mex_muns = regionmask.Regions(list(geo_df['geometry']),names=list(geo_df[geo_col]),
                             abbrevs=list(geo_df[geo_col]),name='MX')
    mask = mex_muns.mask(dataarray,lon_name='x',lat_name='y',wrap_lon=False)
    mask = mask.transpose('y','x')
    dataarray[mask_col] = mask
    if muncode != None:
        dataarray_mask = dataarray.where(mask == list(geo_df[geo_col]).index(str(muncode)), drop=True)
        return dataarray_mask
    else:
        return dataarray


def return_date_id(image_ids):
    ### Takes a satellite image id (or list of ids), which contains the day, month, and year
    ### in the last part of the id and returns a sorted list of those dates as strings
    dates, datestrs = [],[]
    for i, img_id in enumerate(image_ids):
        parts = img_id.split('_')[0]
        year,month, day = parts[:4], parts[4:6], parts[6:]
        dates.append(int(str(year)+str(month)+str(day)))
        if eng_or_esp == "eng":
            datestrs.append(str(month)+'/'+str(day)+'/'+str(year))
        else:
            datestrs.append(str(day)+'/'+str(month)+'/'+str(year))
    return datestrs[dates.index(sorted(dates, reverse=True)[0])]

def plot_image(input_datarray,minx,maxx,miny,maxy,mun_code,
               band_start=0,band_end=3,vmax='',vmin='',
               munname='',date='',subtitle='',colorbar=True,
               axes=False, box_id='box', save_to='', masked=False,
               analysis_type='muns',state_dict={}):
    ### plot_image -- Takes an input_datarray (i.e. xarray image) and performs custom defaults so we don't
    ### need to repeat them.
    ### - minx, maxx, miny, maxy are necessary to provide but these are just the lat/lon
    ### coordinates (i.e.EPSG:4326 projection) of the satellite image/box, which will not be the coordinates
    ### that xarray would otherwise provide (i.e. the coordinates would be projected in EPSG:32416 for example)
    ### so that if you plot axes=True, these axes are in lat/lon coordinates.
    ### - vmax and vmin specify the maximum and minimum values of the matrix we wish to plot, if provided
    ### - munname plots the municipality name
    ### - date plots the data
    ### - box_id is a necessary input to know which folder to write output to
    ### - save_to, if modified, edits the output file name

    fig, ax = plt.subplots()
    if band_start != band_end:
        im_shp = np.moveaxis(input_datarray.data,0,-1)
        if vmin == '':
            im = ax.imshow(im_shp[:,:,band_start:band_end],
                 extent=(minx,maxx,miny,maxy))
        else:
            im = ax.imshow(im_shp[:,:,band_start:band_end],
                 extent=(minx,maxx,miny,maxy),
                          vmin=vmin,vmax=vmax)
    else:
        if vmin == '':
            im = ax.imshow(input_datarray,
                 extent=(minx,maxx,miny,maxy))
        else:
            im = ax.imshow(input_datarray,
                 extent=(minx,maxx,miny,maxy),
                          vmin=vmin,vmax=vmax)
    if munname != '':
        if eng_or_esp == "eng":
            if mex_or_mmr == "mex":
                plt.title('Municipality of '+munname+', '+str(np.round(maxy,2))+'N, '+str(np.round(maxx,2))+'W '+str(date), fontsize=10)
            else:
                plt.title(munname+' Township, '+str(np.round(maxy,2))+'N, '+str(np.round(maxx,2))+'E '+str(date), fontsize=10)
        else:
            if mex_or_mmr == "mex":
                plt.title('Municipio de '+munname+', '+str(np.round(maxy,2))+'N, '+str(np.round(maxx,2))+'W '+str(date), fontsize=10)
            else:
                plt.title('Municipio de '+munname+', '+str(np.round(maxy,2))+'N, '+str(np.round(maxx,2))+'E '+str(date), fontsize=10)
    if subtitle != '':
        plt.suptitle(subtitle, fontsize=14)
    if colorbar == True:
        fig.colorbar(im, ax=ax)
    if axes==False:
        ax.axis('off')
    if save_to != '':
        if analysis_type == 'muns':
            output_fl_dir = os.path.join(remote_sen_output_dir,str(mun_code).replace(str(mun_code)[-3:],""),str(mun_code),box_id,eng_or_esp)
            if not os.path.exists(output_fl_dir):
                os.makedirs(output_fl_dir)
            plt.savefig(os.path.join(output_fl_dir,ciclo+yr+"_"+str(box_id)+"_"+save_to),pad_inches=0.4, dpi=600)

            if masked:
                secondoutputdir = os.path.join(remote_sen_output_dir,str(mun_code).replace(str(mun_code)[-3:],""),str(mun_code),"Masked",ciclo+yr,"no")
                if not os.path.exists(secondoutputdir):
                    os.makedirs(secondoutputdir)
                plt.savefig(os.path.join(secondoutputdir,ciclo+yr+"_"+str(box_id)+".png"),pad_inches=0.4, dpi=600)
        else:
            output_fl_dir = os.path.join(remote_sen_output_dir,str(state_dict[mun_code]),str(mun_code),box_id,eng_or_esp)
            if not os.path.exists(output_fl_dir):
                os.makedirs(output_fl_dir)
            plt.savefig(os.path.join(output_fl_dir,grp+"_"+str(box_id)+"_"+save_to),pad_inches=0.4, dpi=600)

            if masked:
                secondoutputdir = os.path.join(remote_sen_output_dir,str(state_dict[mun_code]),str(mun_code),"Masked",grp,"no")
                if not os.path.exists(secondoutputdir):
                    os.makedirs(secondoutputdir)
                plt.savefig(os.path.join(secondoutputdir,grp+"_"+str(box_id)+".png"),pad_inches=0.4, dpi=600)

    plt.show()

def create_box_geodf(xcenter, ycenter, municipality_code, delta = 0.008333333, epsg=4326):
    ### The file "mun_boxes_'+ciclo+'_start_'+yr+'.csv" tells us about all of the relevant AOIs/boxes for a given
    ### municipality. In this file, each "box" is just a lat/lon coordinate representing the center of the box.
    ### What we need to subset down imagery to each box is to create a geopandas dataframe where the box
    ### dimensions are a polygon. This returns a polygon of said box.
    ### xcenter, ycenter -- lat/lon representing center of the box
    ### municipality_code -- the municipality ID of interest
    ### delta -- the height/weight of boxes, don't change this unless you change the size of boxes in
    ### create_suitability_index.py
    ytop, ybottom = ycenter+(delta/2.0), ycenter-(delta/2.0)
    xleft, xright = xcenter-(delta/2.0), xcenter+(delta/2.0)
    box_poly = Polygon([(xleft, ytop), (xright, ytop), (xright, ybottom), (xleft, ybottom)])
    box_poly_inv = Polygon([(ytop, xleft), (ytop, xright), (ybottom, xright), (ybottom, xleft)])

    ### We need to account for how geopandas flips indices between version 0.5 and 0.6
    if version.parse(gpd.__version__) >= version.parse("0.6"):
        poly_df = gpd.GeoDataFrame({'geometry':[box_poly]})
        poly_df.crs = {'init' :'EPSG:'+str(epsg)}
    else:
        poly_df = gpd.GeoDataFrame({'geometry':[box_poly_inv]})
        poly_df.crs = "EPSG:"+str(epsg)

    poly_df['muncode'] = str(municipality_code)
    return poly_df, ytop, ybottom, xleft, xright

def check_geo_proj_matching(image1_crses,image2_crses):
    ### check_geo_proj_matching.py -- For analysis, we need to ensure that both sets of (averaged) images we
    ### are working with are in the same projection. This is because xarray can only perform operations like
    ### addition and subtraction if the lat/lon coordinates are the same (and you'd need to be smart about
    ### reprojecting/interpolating the image otherwise). Therefore, it looks at the crses (i.e. a string
    ### containing info about the geographic projection of an image) of both the start and end image of the
    ### season, and ensures that they are matching. Throws an error otherwise. If no error thrown, returns
    ### the crs for the imagery so that it can be saved later

    ### Checks that we only have one CRS for each potential set of images we're working with
    if len(image1_crses) == 1:
        pass
    elif len(image2_crses) == 1:
        pass
    else:
        return False, ''

    ### Checks that the CRS for the fallow and crop image is the same, and stores this for posterity
    if len(image1_crses) > 0 and len(image2_crses) > 0:
        if image1_crses[0] != image2_crses[0]:
            return False, ''
        else:
            crs_image = image1_crses[0].split('epsg:')[-1]
            return True, crs_image
    else:
        return False, ''

def compute_histograms(crop_datarray,fallow_datarray,
                       adc07dict, adc16dict,
                       ciclo_yr='sereno_start_17', red_i=2, nir_i=3, nbands=25):
    crop_array = np.nan_to_num(np.array(crop_datarray.data), nan=-1)
    fallow_array = np.nan_to_num(np.array(fallow_datarray.data), nan=-1)

    red_band_crop = crop_array[red_i,:,:]
    nir_band_crop = crop_array[nir_i,:,:]
    red_band_fallow = fallow_array[red_i,:,:]
    nir_band_fallow = fallow_array[nir_i,:,:]

    condensed_info = list(zip(red_band_crop.ravel(),nir_band_crop.ravel(),red_band_fallow.ravel(),
                              nir_band_fallow.ravel(),crop_datarray.adc07.data.ravel(), crop_datarray.adc16.data.ravel()))
    condensed_info = [info for info in condensed_info if np.sum(info) >= 0]
    unique_adc07s = list(set([info[4] for info in condensed_info]))
    unique_adc16s = list(set([info[5] for info in condensed_info]))
    list_of_adc07_hists, list_of_adc16_hists = [], []

    for adc in unique_adc07s:
        hist_total = np.zeros((nbands,nbands,nbands,nbands))
        for rbc, nbc, rbf, nbf, adc07, adc16 in condensed_info:
            if adc == adc07:
                hist_4_pos = hist_to_matrix_pos(rbc),hist_to_matrix_pos(nbc),hist_to_matrix_pos(rbf),hist_to_matrix_pos(nbf)
                if None not in hist_4_pos:
                    hist_total[hist_4_pos] += 1
        list_of_adc07_hists.append([hist_total,adc07dict[adc],ciclo_yr])

    for adc in unique_adc16s:
        hist_total = np.zeros((nbands,nbands,nbands,nbands))
        for rbc, nbc, rbf, nbf, adc07, adc16 in condensed_info:
            if adc == adc16:
                hist_4_pos = hist_to_matrix_pos(rbc),hist_to_matrix_pos(nbc),hist_to_matrix_pos(rbf),hist_to_matrix_pos(nbf)
                if None not in hist_4_pos:
                    hist_total[hist_4_pos] += 1
        list_of_adc16_hists.append([hist_total,adc16dict[adc],ciclo_yr])
    return list_of_adc07_hists, list_of_adc16_hists

def hist_to_matrix_pos(input_val, min_val=0, max_val=0.6, nbands=25):
    if min_val == 0:
        if max_val == 1:
            bins_hist = np.array(list(np.round(np.linspace(0, 1, nbands+1),3)))
        else:
            bins_hist = np.array(list(np.round(np.linspace(0, max_val, nbands),3))+[1.0])
    else:
        if max_val == 1:
            bins_hist = np.array([0.0]+list(np.round(np.linspace(min_val, 1, nbands),3)))
        else:
            bins_hist = np.array([0.0]+list(np.round(np.linspace(min_val, max_val, nbands-1),3))+[1.0])
    hist_vals = np.histogram(input_val, bins_hist)[0]
    index = np.where(hist_vals==1)[0]
    if len(index) == 0:
        return None
    else:
        return index[0]

def total_img_hist(list_of_img_hist, nbands=25, adc_level=True):
    """Takes in a list of image histograms of same dimension and returns a combined image histogram.
    Assumes that bins and dimensions for input image histograms are the same."""
    unique_cicloyrs = list(set([this_hist[2] for this_hist in list_of_img_hist]))
    if adc_level:
        unique_adcs = list(set([this_hist[1] for this_hist in list_of_img_hist]))
    else:
        unique_adcs = list(set([this_hist[1][:5] for this_hist in list_of_img_hist]))

    list_of_hists = []
    for cicloyr in unique_cicloyrs:
        for adc in unique_adcs:
            hist_total = np.zeros((nbands,nbands,nbands,nbands))
            for hist in list_of_img_hist:
                if adc_level:
                    area = hist[1]
                else:
                    area = hist[1][:5]
                if adc == area:
                    if cicloyr == hist[2]:
                        hist_total += hist[0]
            list_of_hists.append([hist_total,adc,cicloyr])
    return list_of_hists
