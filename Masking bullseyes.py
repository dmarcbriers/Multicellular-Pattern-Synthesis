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
    ECAD_mask_dir = pathlib.Path('/Users/Ashley/Documents/UCSF/EBICS/InCell B (one channel stitch)/ECAD/')
    DAPI_dir = pathlib.Path('/Users/Ashley/Documents/UCSF/EBICS/InCell B (one channel stitch)/DAPI/')
else:
    ECAD_mask_dir = pathlib.Path(sys.argv[1])
    DAPI_dir = ECAD_mask_dir

bullseye_file = ECAD_mask_dir.parent/'Bullseye Parameters TESTING!!!(incell).csv'
tot_bullseye_file = ECAD_mask_dir.parent/'Tot Bullseye Parameters TESTING!!!(incell).csv'



DAPI_MIN_DIST= 7
DAPI_THRESHOLD_ABS = 20
SHOW_PLOTS = True
SHOW_PLOTS_DAPI = False
ANALYZE = True
HISTOGRAM = False
# change to True when things stop working to figure where it went wrong

Stored_colony_center = []
Stored_colony_area = []
def segment_colonies(Island_FP, Total_FP, ECAD_mask_file, DAPI_file):
    
    ECAD_mask = imread(str(ECAD_mask_file))#mask_flie are all the WT masks
    ECAD_mask = rgb2grey(ECAD_mask) > 225
#    ECAD_mask = ECAD_mask > 0.2
    ECAD_mask = remove_small_objects(ECAD_mask, min_size = 100)
    ECAD_mask = remove_small_holes(ECAD_mask, min_size = 200)
    ECAD_mask = binary_dilation(ECAD_mask)
    if SHOW_PLOTS:
        plt.imshow(ECAD_mask)
        plt.show()
    #this is creating the mask, first convert to grey scale from rgb, then threshold, then remove holes
    
    
    DAPI = imread(str(DAPI_file))# all the DAPI files
    #imread reads in images
#    print(ECAD_mask_file)
#    plt.imshow(ECAD_mask) 
#    plt.show()
    DAPI = rgb2grey(DAPI).astype(np.float)
    DAPI_mask = DAPI > 139
    
    DAPI_mask = remove_small_objects(DAPI_mask, min_size = 100000)
    DAPI_mask = remove_small_holes(DAPI_mask, min_size = 2000)
    print(DAPI_file)
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
        ax3.imshow(DAPI_labels, cmap = 'inferno')
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
#        Stored_colony_center.append(colony_center)
        # record the measurements in a list is what append does. it adds to list
    '''Counting total number of islands in colony'''
      
    '''Counting total DAPI cells in colony'''   
      
#    DAPI_peaks_tot = peak_local_max(DAPI, threshold_abs = .04, indices = True, min_distance = 5 )
#    # calculate total cells by DAPI image
#    DAPI_peaks_tot.shape[0]
     #number is points or in this case cells
#    print('TOTAL DAPI',DAPI_peaks_tot.shape[0])
#    print('TOTAL ISLANDS', np.unique(labels).shape[0] - 2) #number of islands. np.unique returns and array in 1 dimention and we want to know the shape of the array so add .shap[0]
#    
#    '''calc mean island cells'''
#    np.mean(Stored_DAPI_peaks)
##    print('MEAN cells in island', np.mean(Stored_DAPI_peaks))    
#
#    '''calc STDEV island cells'''
#    np.std(Stored_DAPI_peaks)
##    print('STDEV cells in island', np.std(Stored_DAPI_peaks))   
#
#    '''calc mean island area'''
#    np.mean(Stored_DAPI_peaks)
##    print('MEAN island area', np.mean(Stored_island_area))    
#
#    '''calc STDEV island area'''
#    np.std(Stored_DAPI_peaks)
##    print('STDEV island area', np.std(Stored_island_area)) 
#    
#    '''calc mean island perimeter'''
#    np.mean(Stored_DAPI_peaks)
##    print('MEAN island perimeter', np.mean(Stored_island_perimeter))    
#
#    '''calc STDEV island perimeter'''
#    np.std(Stored_DAPI_peaks)
##    print('STDEV island perimeter', np.std(Stored_island_perimeter)) 
#
#    '''calc mean island eccentricity'''
#    np.mean(Stored_DAPI_peaks)
##    print('MEAN island ecc', np.mean(Stored_island_eccentricity))    
#
#    '''calc STDEV island eccentricity'''
#    np.std(Stored_DAPI_peaks)
##    print('STDEV island ecc', np.std(Stored_island_eccentricity))
#
#    '''calc mean island circularity'''
#    np.mean(Stored_island_circ)
#    print('MEAN island circ', np.mean(Stored_island_circ))
#    
#    '''calc STDEV island circularity'''
#    np.std(Stored_island_circ)
##    print('STDEV island circ', np.std(Stored_island_circ))


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


ECAD_mask_files = {}
DAPI_files = {}

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
                first_term = ECAD_mask_file.name.split('(',1)
                first_term = first_term[0]
                if ECAD_mask_file.name.endswith('(fld 1 wv DAPI - DAPI).tif'):
                    DAPI_files[first_term] = ECAD_mask_file
                if ECAD_mask_file.name.endswith('(fld 1 wv Cy5 - Cy5).tif'):
                    ECAD_mask_files[first_term] = ECAD_mask_file
                    
            for ECAD_mask_file in DAPI_dir.iterdir():
                first_term = ECAD_mask_file.name.split('(',1)
                first_term = first_term[0]
                if ECAD_mask_file.name.endswith('(fld 1 wv DAPI - DAPI).tif'):
                    DAPI_files[first_term] = ECAD_mask_file
                if ECAD_mask_file.name.endswith('(fld 1 wv Cy5 - Cy5).tif'):
                    ECAD_mask_files[first_term] = ECAD_mask_file
                    
            for first_term in ECAD_mask_files:
                ECAD_mask_file = ECAD_mask_files[first_term]
                DAPI_file = DAPI_files[first_term]
                #changed this to use on the RNAseq computer
                if not DAPI_file.is_file():
                    print ('No DAPI_file')
                    print(DAPI_file)
                    print(ECAD_mask_file)
                # to match right dapi to mask we take the DAPI directory and find the file that matches the name of the mask file just with a C3 instead of a C1
        #        print(mask_file, DAPI_file)
        #        print(mask_file.is_file()) #to test that these are files. if not TRUE then something is wrong
        #        print(DAPI_file.is_file())
                segment_colonies(Islands_FP, Total_FP, ECAD_mask_file, DAPI_file) 
                
    #        for mask_file in maskdir.iterdir():
    #            DAPI_file = DAPIdir/mask_file.name.replace('c1.tif','c3.tif')
    #            # to match right dapi to mask we take the DAPI directory and find the file that matches the name of the mask file just with a C3 instead of a C1
    #    #        print(mask_file, DAPI_file)
    #    #        print(mask_file.is_file()) #to test that these are files. if not TRUE then something is wrong
    #    #        print(DAPI_file.is_file())
    #            analyze_islands(Islands_FP,Total_FP, str(mask_file), str(DAPI_file))
    #        #calling the defined function --> analyze_files
    
    
'''Making violin plot'''


RC_PARAMS_DARK = {
    'figure.titlesize': '32',
    'figure.facecolor': 'black',
    'figure.edgecolor': 'white',
    'text.color': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial, Liberation Sans, Bitstream Vera Sans, sans-serif',
    'axes.labelcolor': 'white',
    'axes.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.linewidth': '2',
    'axes.axisbelow': 'True',
    'axes.grid': 'False',
    'axes.titlesize': '24',
    'axes.labelsize': '20',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'xtick.major.size': '3',
    'ytick.major.size': '3',
    'xtick.minor.size': '3',
    'ytick.minor.size': '3',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'legend.frameon': 'False',
    'legend.numpoints': '1',
    'legend.scatterpoints': '1',
    'legend.fontsize': '20',
    'image.cmap': 'Greys',
    'grid.linestyle': '-',
    'lines.solid_capstyle': 'round',
    'lines.linewidth': '3',
    'lines.markersize': '10',
    'grid.color': 'white'}

RC_PARAMS_LIGHT = {
    'figure.titlesize': '32',
    'figure.facecolor': 'white',
    'figure.edgecolor': 'black',
    'text.color': 'black',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial, Liberation Sans, Bitstream Vera Sans, sans-serif',
    'axes.labelcolor': 'black',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': '1.5',
    'axes.spines.left': 'True',
    'axes.spines.bottom': 'True',
    'axes.spines.top': 'False',
    'axes.spines.right': 'False',
    'axes.axisbelow': 'True',
    'axes.grid': 'False',
    'axes.titlesize': '24',
    'axes.labelsize': '20',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.major.size': '5',
    'ytick.major.size': '5',
    'xtick.major.width': '1.5',
    'ytick.major.width': '1.5',
    'xtick.minor.size': '3',
    'ytick.minor.size': '3',
    'xtick.minor.width': '1.5',
    'xtick.minor.width': '1.5',
    'xtick.labelsize': '20',
    'ytick.labelsize': '20',
    'xtick.top': 'False',
    'xtick.bottom': 'True',
    'ytick.left': 'True',
    'ytick.right': 'False',
    'legend.frameon': 'False',
    'legend.numpoints': '1',
    'legend.scatterpoints': '1',
    'legend.fontsize': '20',
    'image.cmap': 'Greys',
    'grid.linestyle': '-',
    'lines.solid_capstyle': 'round',
    'lines.linewidth': '5',
    'lines.markersize': '10',
    'grid.color': 'black'}

if HISTOGRAM:
    df = pd.read_excel('/Users/Ashley/Documents/UCSF/EBICS/InCell B (one channel stitch)/simulation_metrics_bullseye_02142018.xlsm',
                  usecols= range(24,25))
    
    #print(df)
    #
    #with plt.style.context({'lines.color': 'black', 'legend.fontsize':'large', 'axes.grid': False, 'axes.facecolor': 'white', 'axes.labelweight' : 'heavy', 'axes.labelpad': 7.0, 'axes.labelsize':16, 'xtick.labelsize': 14,'ytick.labelsize': 14 }):
    #    ax = sns.violinplot(data=df, palette = 'muted', inner='quartile')
    #
    #with plt.style.context({'lines.color': 'black', 'legend.fontsize':'large', 'axes.grid': False, 'axes.facecolor': 'white', 'axes.labelweight' : 'heavy', 'axes.labelpad': 7.0, 'axes.labelsize':16, 'xtick.labelsize': 14,'ytick.labelsize': 14 }):
    #    ax = sns.boxplot(data=df, palette = 'muted') #inner='quartile')
    ##this is for a box plot
    
    with plt.style.context(RC_PARAMS_LIGHT):
        ax = sns.distplot(df['Total Clusters red'], bins = 18, color= "r")
        ax.set_xlim(left = 0, right = 16)
        ax.set_ylim(bottom = 0, top = 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig = plt.gcf()
        fig.savefig('/Users/Ashley/Documents/UCSF/EBICS/InCell B (one channel stitch)/Total B sim Islands.svg', transparent = True)
        #histogram for single condition
