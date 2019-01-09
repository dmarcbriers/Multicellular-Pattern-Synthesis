#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:13:37 2018

@author: Ashley
"""
import sys
import numpy as np
from skimage.io import imread #this reads in images
import pandas as pd
from scipy import ndimage as ndi
import seaborn as sns
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt #this lets me plot graphs
from skimage.color import rgb2grey # this converts rgb images to 8bit
from skimage.morphology import label, watershed  # this labels each connected region
from skimage.morphology import remove_small_objects, remove_small_holes, binary_dilation, disk # enssentially smoothens things
from skimage.feature import peak_local_max
from skimage.measure import regionprops #measures the max threshold  within certian radius and records its location
# find contours in skimage google this for circularity
import pathlib
from matplotlib.ticker import MaxNLocator


if len(sys.argv) == 1:
    ECAD_mask_dir = pathlib.Path.home() / 'Documents' / 'Git Hub' / 'Multicellular-Pattern-Synthesis' / 'image_segmentation_clustering' / 'images' / 'bullseye'
else:
    ECAD_mask_dir = pathlib.Path(sys.argv[1])

bullseye_file = ECAD_mask_dir.parent.parent / 'BullseyeParameters.csv'
tot_bullseye_file = ECAD_mask_dir.parent.parent / 'TotBullseyeParameters.csv'

DAPI_MIN_DIST = 7
DAPI_THRESHOLD_ABS = 20
SHOW_PLOTS = True
SHOW_PLOTS_DAPI = True
ANALYZE = True

DAPI_COLOR = (0, 0, 250)
ECAD_COLOR = (255, 0, 0)

# change to True when things stop working to figure where it went wrong

Stored_colony_center = []
Stored_colony_area = []

def segment_colonies(Island_FP, Total_FP, ECAD_mask_file):

    # Load and split the ECAD mask file
    print('Loading colony data from {}'.format(ECAD_mask_file))
    raw_mask = imread(str(ECAD_mask_file))
    if raw_mask.ndim != 3 or raw_mask.shape[2] != 3:
        raise ValueError('Expected color image for {} got {}'.format(ECAD_mask_file, raw_mask.shape))

    ecad_weights = np.array(ECAD_COLOR).reshape((1, 1, 3)) / np.sum(ECAD_COLOR)
    ECAD_mask = np.sum(raw_mask * ecad_weights, axis=2) > 75

    #    ECAD_mask = ECAD_mask > 0.2
    ECAD_mask = remove_small_objects(ECAD_mask, min_size = 100)
    ECAD_mask = remove_small_holes(ECAD_mask, min_size = 2000)
    ECAD_mask = binary_dilation(ECAD_mask)
    if SHOW_PLOTS:
        plt.imshow(ECAD_mask)
        plt.show()

    #this is creating the mask, first convert to grey scale from rgb, then threshold, then remove holes
    dapi_weights = np.array(DAPI_COLOR).reshape((1, 1, 3)) / np.sum(DAPI_COLOR)
    DAPI = np.sum(raw_mask * dapi_weights, axis=2)
    DAPI_mask = DAPI > 10

    DAPI_mask = remove_small_objects(DAPI_mask, min_size = 100)
    DAPI_mask = remove_small_holes(DAPI_mask, min_size = 20000)
    
    if SHOW_PLOTS:
        plt.imshow(DAPI_mask)
        plt.show()
        
    DAPI_dist = ndi.distance_transform_edt(DAPI_mask)
    local_maxi = peak_local_max(DAPI_dist, indices=False,min_distance = 500, labels=DAPI_mask) # if you want to include border images use exclude_border = False
    markers = ndi.label(local_maxi)[0]
    DAPI_labels = watershed(-DAPI_dist, markers, mask = DAPI_mask )+1
    DAPI+= np.random.ranf(DAPI.shape)*1e-5
    #region props ingnores everything equal to zero so watershed returns labels [0,1,2,3...] regio props ignores the first one, which is stupid
#    DAPI = binary_dilation(DAPI, disk(100)) # slows down computer too much
    #larger scale than binary_dilation the disk is the rasdius of ball rolling to fill gaps
    if SHOW_PLOTS:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(DAPI)
        ax2.imshow(DAPI_dist, cmap='inferno')
        ax3.imshow(DAPI_labels, cmap='inferno')
        plt.show()
    print(np.unique(DAPI_labels))
#    plt.imshow(np.logical_and(ECAD_mask, DAPI_labels))
#    plt.show()
    largest_colony_ID = []

    for colony_ID, region in zip(np.unique(DAPI_labels),regionprops(DAPI_labels)):
        largest_colony_ID.append(region.area)
        print(largest_colony_ID)
    for colony_ID, region in zip(np.unique(DAPI_labels), regionprops(DAPI_labels)):
        if colony_ID == np.argmax(largest_colony_ID)+1:
            print('skipping id, {}'.format(colony_ID))
            continue #this is to skip the largest segment form watershed

        colony_center = region.centroid
        colony_area = region.area

        DAPI_colony = DAPI.copy()
        DAPI_colony[DAPI_labels != colony_ID] = 0
        DAPI_peaks_tot = peak_local_max(DAPI_colony, threshold_abs = DAPI_THRESHOLD_ABS, indices = True, min_distance = DAPI_MIN_DIST)
        DAPI_peaks_tot.shape[0]
        print(DAPI_peaks_tot.shape[0])

        if SHOW_PLOTS_DAPI:

            plt.imshow(DAPI_colony, cmap= 'gist_earth')
            plt.plot(DAPI_peaks_tot[:,1],DAPI_peaks_tot[:,0], '.r')
            plt.xlim(1000,1100)
            plt.ylim(3000,3100)
#            plt.xlim(np.min(DAPI_peaks_tot[:,1]), np.max(DAPI_peaks_tot[:,1]))
#            plt.ylim(np.min(DAPI_peaks_tot[:,0]), np.max(DAPI_peaks_tot[:,0]))
            plt.colorbar()
            plt.show()
        if SHOW_PLOTS_DAPI:

            plt.imshow(DAPI_colony, cmap= 'gist_earth')
            plt.plot(DAPI_peaks_tot[:,1],DAPI_peaks_tot[:,0], '.r')
            plt.xlim(1000,1100)
            plt.ylim(2000,2100)
            plt.colorbar()
            plt.show()
        if SHOW_PLOTS_DAPI:

            plt.imshow(DAPI_colony, cmap= 'gist_earth')
            plt.plot(DAPI_peaks_tot[:,1],DAPI_peaks_tot[:,0], '.r')
            plt.xlim(1000,1100)
            plt.ylim(3500,3600)
            plt.colorbar()
            plt.show()
        if SHOW_PLOTS_DAPI:

            plt.imshow(DAPI_colony, cmap= 'gist_earth')
            plt.plot(DAPI_peaks_tot[:,1],DAPI_peaks_tot[:,0], '.r')
            plt.xlim(1000,1100)
            plt.ylim(3700,3800)
            plt.colorbar()
            plt.show()
#        Island_FP.write('{},{}\n'.format(colony_ID, colony_center))
        if SHOW_PLOTS:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3)
            ax1.imshow(ECAD_mask, cmap = 'inferno')
            ax2.imshow(DAPI_labels == colony_ID, cmap='inferno')
            ax3.imshow(np.logical_and(ECAD_mask, DAPI_labels ==colony_ID), cmap = 'inferno')
            plt.show()

        analyze_islands(Island_FP, Total_FP, colony_ID, colony_area, colony_center, ECAD_mask_file, np.logical_and(ECAD_mask, DAPI_labels == colony_ID), DAPI, DAPI_peaks_tot)


def analyze_islands(Islands_FP,Total_FP, colony_ID, colony_area, colony_center,ECAD_mask_file, ECAD_mask, DAPI, DAPI_peaks_tot):
    '''Making the mask over islands and DAPI'''
#making large function so that we can call in each file
#Islands_FP is the eventual writing file function where everything is to be stored


    '''Putting Mask over DAPI image and counting cells in each island'''

    labels = label(ECAD_mask) # means label ECAD parts
    Stored_DAPI_peaks = []
    Stored_island_area = []
    Stored_island_perimeter =[]
    Stored_island_eccentricity = []
    Stored_island_circ = []
    Stored_island_center = []
    Stored_dist_center_norm = []
    Over50_islands = 0 #this is a counter

    for label_ID, region in zip(np.unique(labels[labels>0]),regionprops(labels)):
        #zip takes two lists and zips them together, outputs from one and the other both get put into loop. stops whichever one ends first
#        if label_ID == 0 or label_ID == 1:
#            continue
        #for every unique label if the label is 0 which means the background, which is not the mask, but we still dont want it, or the first one which was edge effects

        #for every unique label if the label is 0 which means the background, which is not the mask, but we still dont want it, or the first one which was edge effects
        labels == label_ID
        #in this loop you are testing (==) whether the label is some label ID, so we can skip ones we don't want
        DAPI_island = DAPI.copy() # want to make a copy so we don't so this to the original image
        DAPI_island[labels != label_ID] = 0 # for the

        DAPI_peaks = peak_local_max(DAPI_island, threshold_abs = DAPI_THRESHOLD_ABS, indices = True, min_distance = DAPI_MIN_DIST )
         #taking the maxium of the DAPI locations and then asigning peaks a value, using rounded DAPI to our advantage
        # indices are the coordinates in the image, so True gives back actual position
        DAPI_peaks.shape[0]
        # need 0 cause that is the first column in matrix and first colum tells you number of points
        if DAPI_peaks.shape[0] < 50:
            continue
#        if region.area > 100000:
#            continue
        if region.area < 10:
            continue
#        if region.perimeter > 10000:
#            continue

#        convex_region = regionprops(region.convex_image.astype(np.uint8))[0]
        Over50_islands += 1 # this can count up number of elements that pass the loop
        print('island{} in {}'.format(Over50_islands, colony_ID))
        dist_from_center_norm = dist.euclidean(colony_center, region.centroid)/ colony_area
        island_circ = (4 * np.pi * region.area)/ (region.perimeter * region.perimeter)
#        if DAPI_file.name.endswith('_s18c3.tif'):
##        if DAPI_file.is_file('/Users/Ashley/Documents/UCSF/EBICS/EBICS predicted patterns/Island patterns/DAY 4/un-edited/DAPI/CDH1 WT d4-Image Export-08_s18c3.tif'):
#           #didn't work error 'str' object has no attribute 'is_file'
        if SHOW_PLOTS:
            Xmin = np.min(DAPI_peaks[:,1])
            Xmax = np.max(DAPI_peaks[:,1])
            Ymin = np.min(DAPI_peaks[:,0])
            Ymax = np.max(DAPI_peaks[:,0])
            # this is just to create MAX to MIN coordinates to help zoom in later
            plt.imshow(DAPI_island, cmap = 'gist_earth')
            plt.plot(DAPI_peaks[:,1],DAPI_peaks[:,0],'.r')
            #peaks is going to be a list of points so we are pulling the x and y coordinate out of peaks
            # normally draws lins, but '.r' plots red dots
            plt.xlim([Xmin,Xmax])
            plt.ylim([Ymin,Ymax])
            #zooming in on plot to Xmin and max
    #
            plt.colorbar()
                # you want this!
            plt.show()
#            print(DAPI_file)
        Islands_FP.write('{},{},{},{}, {}, {}, {},{},{},{}\n'.format(
                ECAD_mask_file,
                colony_ID,
                label_ID,
                DAPI_peaks.shape[0],
                region.area,
                region.perimeter,
                region.eccentricity,
                island_circ,
                region.centroid,
                dist_from_center_norm))
       # write = writing to opened file, curlies are the labels of each, \n is new line, .format takes values and puts them in the curlies string
#        print('LABEL_ID', label_ID)
#        print('CELLS per ISLAND',DAPI_peaks.shape[0])
#        print('AREA',region.area)
#        print('PERIMETER', region.perimeter)
#        print('ECCENTRICITY', region.eccentricity)
#        print('CIRCULARITY', island_circ)
        Stored_DAPI_peaks.append(DAPI_peaks.shape[0])
        Stored_island_area.append(region.area)
        Stored_island_perimeter.append(region.perimeter)
        Stored_island_eccentricity.append(region.eccentricity)
        Stored_island_circ.append(island_circ)
        Stored_island_center.append(region.centroid)
        Stored_dist_center_norm.append(dist_from_center_norm)

    Total_FP.write('{ECAD_mask_file},{colony_ID},{tot_isl},{tot_DAPI},{Mean_cells},{STDEV_cells},{Mean_area},{STDEV_area},{Mean_per},{STDEV_per},{Mean_ecc},{STDEV_ecc},{Mean_circ},{STDEV_circ},{Mean_centroid_isl}, {STDEV_centroid_isl}, {Mean_dist_center_norm}, {STDEV_dist_center_norm}\n'.format(
            ECAD_mask_file = ECAD_mask_file,
            colony_ID = colony_ID,
            tot_isl= Over50_islands,
            tot_DAPI=DAPI_peaks_tot.shape[0],
            Mean_cells = np.mean(Stored_DAPI_peaks),
            STDEV_cells = np.std(Stored_DAPI_peaks),
            Mean_area = np.mean(Stored_island_area),
            STDEV_area = np.std(Stored_island_area),
            Mean_per = np.mean(Stored_island_perimeter),
            STDEV_per = np.std(Stored_island_perimeter),
            Mean_ecc = np.mean(Stored_island_eccentricity),
            STDEV_ecc = np.std(Stored_island_eccentricity),
            Mean_circ = np.mean(Stored_island_circ),
            STDEV_circ = np.std(Stored_island_circ),
            Mean_centroid_isl = np.mean(Stored_island_center),
            STDEV_centroid_isl = np.std(Stored_island_center),
            Mean_dist_center_norm = np.mean(Stored_dist_center_norm),
            STDEV_dist_center_norm = np.std(Stored_dist_center_norm)))

Stored_DAPI_peaks = []
Stored_island_area = []
Stored_island_perimeter =[]
Stored_island_eccentricity = []
Stored_island_circ = []
Stored_dist_center_norm = []
Stored_island_center = []

if ANALYZE:
    with bullseye_file.open('wt') as Islands_FP:
    # this is to open the file where all the generated info is now going to be stored
    # open opens files for reading by default, wt says instead open the file for writing as text (w = writing, t = text)
    # IslAnd_FP can write in a simialr way to print it just goes to the created file instead
        with tot_bullseye_file.open('wt') as Total_FP:
            Total_FP.write('file name, Colony ID, Total Islands, Total DAPI,Mean cells, STDEV cells, Mean Area, STDEV Area, Mean Perimeter, STDEV Perimeter, Mean Eccentricity, STDEV Eccentricy, Mean Circ, STDEV Circ, Mean Centroid, nan , STDEV Centroid, Mean Dist from center, STDEV dist center\n')

         #this is separate file for colony based things
            Islands_FP.write('file name, Label_ID, Cells per Island, Area, Perimeter, Eccentricity, Circ, Centroid, nan, dist from center\n')
        #this is to label each column with \n to start a new line. the string is literally writing down to file

            for ECAD_mask_file in ECAD_mask_dir.iterdir():
                if ECAD_mask_file.name.startswith('EBICS_BULLSEYE_') and ECAD_mask_file.suffix == '.tif':
                    segment_colonies(Islands_FP, Total_FP, ECAD_mask_file)
