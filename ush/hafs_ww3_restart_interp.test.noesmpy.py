#! /usr/bin/env python3
################################################################################
# Script Name: hafs_ww3_restart_interp.test.noesmpy.py
# Author: Ali Salimi-Tarazouj, NCEP/EMC WAVE MODELING TEAM 
#       : modified by Keston Smith to not utilize esmpy
# Abstract:
#   This script interpolates a netcdf WW3 restart to another grid, using ESMPy
# History:
#   08/26/2025: Added the script for interpolating WW3 unstructured netcdf restart file
# Usage:
#       mpirun -n 1 python hafs_ww3_restart_interp.py \
#        --restart_file <restart.nc> \
#        --src_scrip <src_scrip.nc> \
#        --dst_scrip <dst_scrip.nc> \
#        --mask_file <ww3_grid_a.msk> \
#        --output_file <output.nc>
#
#where restart_file is unstructured WW3 restart file, src_script is unstructured mesh
#scrip file, dst_scrip is destination mesh scrip file (e.eg.,hafs mesh), mask_file is
#destination mesh mask file, and output_file is the name of destination restart file
#
# The scrip files are created by running ww3_grid with the SCRIPNC switch
#
# Note, there is a weights file that is created called WHTGRIDINT.nc if you 
# create this file and put it in the run directory, subsequent calls to this script
# will execute faster
#
# hafs_ww3_restart_interp.test.noesmpy.py is a version of hafs_ww3_restart_interp.py
# which does not use esmpy.  Appropriate interpolation weights need to exist in 
# file "WHTGRIDINT.nc" when called.  This was developed to trouble shoot an error 
# believed to be related to esmpy but was in fact related to corrupted input data. 
################################################################################


#import esmpy

import numpy as np
import netCDF4 as nc
import os
import argparse
import time

import xarray as xr
import scipy.sparse as sp

##########################
# === Parse arguments ===#
##########################
parser = argparse.ArgumentParser()
parser.add_argument("--restart_file", required=True)
parser.add_argument("--src_scrip", required=True)
parser.add_argument("--dst_scrip", required=True)
parser.add_argument("--mask_file", required=True)
parser.add_argument("--output_file", required=True)
args = parser.parse_args()

start_time = time.time()


#######################################
# === resuse weights  ===#
#######################################
weights_file = "WHTGRIDINT.nc"
with xr.open_dataset(weights_file) as ds_s:
   # Standard sparse storage uses 'row', 'col', and 'data' variables
   row = ds_s['row'].values
   col = ds_s['col'].values
   weights = ds_s['S'].values

###################################
# === Load and flatten 2D mask ===#
###################################
mask_2d = np.loadtxt(args.mask_file, dtype=int)
nyy, nxx = mask_2d.shape
mask_flat = mask_2d.flatten(order="C")
mask_zero_idx = np.where(mask_flat == 0)[0]
print("maskshape")
print(mask_2d.shape)

##################################################
# === Inject mask into destination SCRIP file ===#
##################################################
print(args.dst_scrip)

nan=float('nan')
#############################
# === Open source NetCDF ===#
#############################
src_nc = nc.Dataset(args.restart_file, "r")
src_nc.set_auto_mask(False)
time_val = src_nc.variables["time"][:]
nt = len(time_val)
nx_src = src_nc.dimensions["nx"].size
nk = src_nc.variables["nk"][:].item()
nth = src_nc.variables["nth"][:].item()

##############################
# === Variables to regrid ===#
##############################
vars_to_regrid = [
  v for v in src_nc.variables
  if ("time" in src_nc.variables[v].dimensions and "nx" in src_nc.variables[v].dimensions)
  and src_nc.variables[v].ndim in (2, 3)
  and (src_nc.variables[v].ndim == 2 or src_nc.variables[v].shape[1] == 1)
]

nx_dst = mask_flat.size

print(str(nx_dst)+" by "+str(nx_src))
# Build SciPy sparse matrix
#matrix = sp.coo_matrix((weights, (row, col)), shape=(nx_dst+1,nx_src)).tocsr()
matrix = sp.coo_matrix((weights, (row-1, col-1)), shape=(nx_dst,nx_src)).tocsr()
print(matrix)

#####################################################
# === Step 1: Regrid and write to temporary file ===#
#####################################################
tmp_file = "interpolated_tmp.nc"
if os.path.exists(tmp_file):
  os.remove(tmp_file)

with nc.Dataset(tmp_file, "w") as out_nc:
  out_nc.set_auto_mask(False)
  out_nc.createDimension("time", nt)
  out_nc.createDimension("ny", 1)
  out_nc.createDimension("nx", nx_dst)
  out_nc.createVariable("time", "f8", ("time",))[:] = time_val
  out_nc.createVariable("nk", "i4")[...] = nk
  out_nc.createVariable("nth", "i4")[...] = nth

  nnn=0
  for varname in vars_to_regrid:
    nnn=nnn+1
    varin = src_nc.variables[varname]
    fill_value = varin.getncattr("_FillValue")
    if (nnn>685) and (nnn< 695):
      print(varin)
    
    if (nnn>685) and (nnn< 695):
      print("varin[:].shape")
      print(varin[:].shape)
    src_data=np.zeros((nt,nx_src))
    src_data[:] = varin[:].reshape(nt, nx_src)

    if (nnn>685) and (nnn< 695):
      print(src_data)

    if varname[0]=='v':
      jfill=np.where(src_data==fill_value)
      src_data[jfill]=nan

    HighLimit=np.max(src_data)
    if nnn<7900:
      print(varname+" High value of input field = ")
      print(HighLimit)

    # 3. Perform Multiplication
    dst_field = matrix @ src_data.T
    jHigh=np.where(dst_field > HighLimit)
    dst_field[jHigh]=HighLimit
    out_var = out_nc.createVariable(varname, "f4", ("time", "ny", "nx"), fill_value=fill_value)

    dst_data=np.zeros((1,1,nx_dst))
    dst_data[0,0,:]=dst_field[:,0]
    out_var[0, 0, :] = dst_data[0,0,:]

src_nc.close()

######################################################
# === Step 2: Apply mask and set mapsta using mask ===
######################################################

with nc.Dataset(tmp_file, "r") as src, nc.Dataset(args.output_file, "w") as dst:
#  src.set_auto_mask(False)
  src.set_auto_mask(True)
  dst.set_auto_mask(False)
  dst.createDimension("time", nt)
  dst.createDimension("ny", nyy)
  dst.createDimension("nx", nxx)
  dst.createVariable("time", "f8", ("time",))[:] = src.variables["time"][:]
  dst.createVariable("nk", "i4")[...] = src.variables["nk"][:]
  dst.createVariable("nth", "i4")[...] = src.variables["nth"][:]

  for varname, varin in src.variables.items():
    if varname in ["time", "nk", "nth"]:
      continue
    if varname == "mapsta":
      fill_value = -2147483647
      data = mask_flat
      data[mask_zero_idx] = fill_value
      reshaped = data.reshape((nt, nyy, nxx), order="C")
      dst_var = dst.createVariable(varname, "i4", ("time", "ny", "nx"), fill_value=fill_value)
      dst_var[:] = reshaped
    else: 
      fill_value = varin.getncattr("_FillValue")
      data = varin[:, 0, :].copy()
      data[:, mask_zero_idx] = fill_value
      reshaped = data.reshape((nt, nyy, nxx), order="C")
      dst_var = dst.createVariable(varname, "f4", ("time", "ny", "nx"), fill_value=fill_value)
      dst_var[:] = reshaped

#Cleanup
#os.remove(tmp_file)
print(f"Done. Output: {args.output_file} | Elapsed: {time.time() - start_time:.1f} sec")

