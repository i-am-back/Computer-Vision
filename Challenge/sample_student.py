## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, classify.py
## 3. In this challenge, you are only permitted to import numpy and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import os, sys
sys.path.append('..')
from util.filters import filter_2d
from util.image import convert_to_grayscale
from numpy import argmin
#It's kk to import whatever you want from the local util module if you would like:
#from util.X import ... 

def classify(im):
    gray = convert_to_grayscale(im/255.)
    Kx = np.array([[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]])

    Ky = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])
    Gx = filter_2d(gray, Kx)
    Gy = filter_2d(gray, Ky)

    #Compute Gradient Magnitude and Direction:
    G_magnitude = np.sqrt(Gx**2+Gy**2)
    G_direction = np.arctan2(Gy, Gx)
    edge_angle_values = G_direction[G_magnitude > 0.95]
    #print (edge_angle_values)
    a = np.histogram(edge_angle_values, bins = 50)
    std_dev = np.std(a[0])
    if std_dev < 7:
        return 'ball'
    #elif std_dev > 7 and std_dev < 15:
    #   return 'cylinder'
    #print (np.std(a[0]))
    edges = G_magnitude > 0.95
    #a = np.histogram(edges,50)
    #std_dev = np.std(a[0])
    #if  std_dev< 7:
    #    return 'ball'
    y_coords,x_coords = np.where(edges)
    y_coords_flipped = edges.shape[0] - y_coords

    phi_bins = 128
    theta_bins = 128
    accumulator = np.zeros((phi_bins, theta_bins))
    rho_min = -edges.shape[0]
    rho_max = edges.shape[1]
    theta_min = 0
    theta_max = np.pi
    #Compute the rho and theta values for the grids in our accumulator:

    rhos = np.linspace(rho_min, rho_max, accumulator.shape[0])
    thetas = np.linspace(theta_min, theta_max, accumulator.shape[1])
    for i in range(len(x_coords)):
        #Grab a single point
        x = x_coords[i]
        y = y_coords_flipped[i]

        #Actually do transform!
        curve_rhos = x*np.cos(thetas)+y*np.sin(thetas)

        for j in range(len(thetas)):
            #Make sure that the part of the curve falls within our accumulator
            if np.min(abs(curve_rhos[j]-rhos)) <= 1.0:
                #Find the cell our curve goes through:
                rho_index = argmin(abs(curve_rhos[j]-rhos))
                accumulator[rho_index, j] += 1
    max_value = np.max(accumulator)
    #print (np.std(edges))
    #print (std_dev)
    #print (std_dev,max_value)
    if max_value < 65 :
        return 'ball'
    elif max_value >130:
        return 'brick'
    else:
        return 'cylinder'