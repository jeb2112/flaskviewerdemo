# script processes misc radnec2 predictions with nnunet 2d model
# to assemble 2d predictions back into 3d files
# doesn't make use of segmentation or support lesion number yet

import os
import json
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial.distance import dice
import imageio
import numpy as np
import nibabel as nb
import pandas as pd
import json
import copy
import argparse
import shutil
import subprocess
import sys

# load a single nifti file
def loadnifti(t1_file,dir,type=None):
    img_arr_t1 = None
    try:
        img_nb_t1 = nb.load(os.path.join(dir,t1_file))
    except IOError as e:
        print('Can\'t import {}'.format(t1_file))
        return None,None
    nb_header = img_nb_t1.header.copy()
    # nibabel convention will be transposed to sitk convention
    img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
    if type is not None:
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine

    return img_arr_t1,affine

# write a single nifti file. use uint8 for masks 
def writenifti(img_arr,filename,header=None,norm=False,type='float64',affine=None):
    img_arr_cp = copy.deepcopy(img_arr)
    if norm:
        img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
    # using nibabel nifti coordinates
    img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
    nb.save(img_nb,filename)
    if True:
        os.system('gzip --force "{}"'.format(filename))

def recycle_dims(alist):
    while True:
        for item in alist:
            yield item

def main(datadir):
    niftidir = os.path.join(datadir,'dicom2nifti_upload')
    predictiondir = os.path.join(datadir,'nnUNet_predictions','flask')
    resultsdir = os.path.join(predictiondir,'results')
    try:
        shutil.rmtree(resultsdir)
    except FileNotFoundError:
        pass
    os.makedirs(resultsdir,exist_ok=True)

    # hard-coded convention from nnunet_predict_preprocess
    olist = [(0,'ax'),(1,'sag'),(2,'cor')]
    orients = recycle_dims(olist)

    cases = sorted(set([re.search('(M|DSC)_?[0-9]*',f)[0] for f in os.listdir(niftidir)]))

    for case in cases:

        print('processing case {}'.format(case))

        preds = sorted([f for f in os.listdir(predictiondir) if case in f])
        studies = sorted(set([re.search('[0-9]{8}',f)[0] for f in preds]))

        for s in studies:

            print('study {}'.format(s))

            # try to load original nifti volume for reference
            # image dim, affine should be saved in a json instead during preprocess script
            imgs_nii = {}
            for ik in ['flair+','t1+','flair','t1']:
                filename = glob.glob(os.path.join(niftidir,case,s,ik+'_processed*'))
                if len(filename):
                    # will use 8 bit now for png, but could be 32bit tiffs
                    imgs_nii[ik],affine = loadnifti(os.path.split(filename[0])[1],os.path.join(niftidir,case,s),type='uint8')
                    image_dim = np.shape(imgs_nii[ik])
                    break # ie processed images are all resampled to same matrix

            pred_3d = {'ax':None,'sag':None,'cor':None,'comp_OR':None,'comp_AND':None}

            for k in pred_3d.keys():
                pred_3d[k] = np.zeros(image_dim)

            orient = next(recycle_dims(orients))
            pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
            # since brains aren't currently being extracted, can't easily test for a non-zero slice 
            # so have to process all slices including air background
            islice = 0
            idimold = image_dim[1:] # ie because the images were originally created and labelled starting with 'ax'
            study_preds = sorted([f for f in preds if s in f])

            for p in study_preds:

                pfile = os.path.join(predictiondir,p)
                pred_arr = imageio.v3.imread(pfile)
                idim = np.array(np.shape(pred_arr))

                if any(idim != idimold):
                        pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
                        output_fname = os.path.join(predictiondir,'experiment1','pred_3d','pred_' + case + '_' + s + '_' + orient[1] + '.nii')
                        if False:
                            writenifti(pred_3d[orient[1]],output_fname,affine=affine)
                        orient = next(recycle_dims(orients))
                        pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],orient[0],0)
                        islice = 0
                        idimold = np.copy(idim)
                else:
                    pass

                if islice >= np.shape(pred_3d[orient[1]])[0]:
                    raise IndexError
                pred_3d[orient[1]][islice] = np.copy(pred_arr)
                islice += 1

            # output the final 'cor' 3d
            pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
            output_fname = os.path.join(predictiondir,'experiment1','pred_3d','pred_' + case + '_' + s + '_' + orient[1] + '.nii')
            if False:
                writenifti(pred_3d[orient[1]],output_fname,affine=affine)

            if True: # output composite 3d
                compT_OR = (pred_3d['ax']==1) | (pred_3d['sag']==1) | (pred_3d['cor']==1)
                compRN_OR = (pred_3d['ax']==2) | (pred_3d['sag']==2) | (pred_3d['cor']==2)
                pred_3d['compOR'] = np.zeros(image_dim)
                pred_3d['compOR'][np.where(compRN_OR)] = 5 
                pred_3d['compOR'][np.where(compT_OR)] = 6 # T overwrites RN
                # lesion number hard-coded here
                output_fname = os.path.join(resultsdir,'pred_' + case + '_' + s + '_1_compOR.nii')
                writenifti(pred_3d['compOR'],output_fname,affine=affine)

        # copy all case nifti files to output directory as well for reference
        for studydir in glob.glob(os.path.join(niftidir,case, '*')):
            shutil.copytree(studydir, os.path.join(resultsdir,os.path.basename(studydir)))

    # create download zip file
    # currently not separated if multiple cases, just named for last case processed
    current_dir = os.getcwd()
    os.chdir(resultsdir)
    command = 'zip -r ' + os.path.join(predictiondir,case+'_inference.zip') + ' *'
    os.system(command)
    os.chdir(current_dir)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--uploaddir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom_upload")
    parser.add_argument("--niftidir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom2nifti_upload")
    parser.add_argument("--datadir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/") 
    args, unknown_args = parser.parse_known_args()
    main(args.datadir)
