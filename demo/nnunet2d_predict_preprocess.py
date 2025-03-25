# script copies 3d processed nifti files to a 2d nnunet file and directory 
# format for nnunet2d inference prediction

import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import cv2
import nibabel as nb
import shutil
import matplotlib.pyplot as plt
from skimage.io import imsave
import glob

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
        if np.max(img_arr_t1) > 255:
            img_arr_t1 = (img_arr_t1 / np.max(img_arr_t1) * 255)
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine
    return img_arr_t1,affine

# main
def main(datadir):

    niidir = os.path.join(datadir,'dicom2nifti_upload')
    # nnunetdir = os.path.join(datadir,'nnUNet_raw','Dataset139_RadNec')
    pred_dir = os.path.join(datadir,'nnUNet_raw','flask','imagesTs')

    cases = sorted(os.listdir(niidir))

    img_idx = 1

    for c in cases:
        print('processing case ' + c)

        output_imgdir = pred_dir
        try:
            shutil.rmtree(output_imgdir)
        except FileNotFoundError:
            pass
        os.makedirs(output_imgdir,exist_ok=True)


        if False: #debugging
            if c != 'M0066':
                continue

        cdir = os.path.join(niidir,c)
        os.chdir(cdir)
        imgs = {}

        studies = os.listdir(cdir)

        for s in studies:
            print('study ' + s)

            imgs = {}
            for ik in ['flair+','t1+']:
                filename = glob.glob(os.path.join(s,ik+'_processed*'))[0]
                # will use 8 bit now for png, but could be 32bit tiffs
                imgs[ik],_ = loadnifti(os.path.split(filename)[1],os.path.join(cdir,s),type='uint8')

            for dim in range(3):
                slices = range(np.shape(imgs[ik])[dim])
                for slice in slices:
                    imgslice = {}
        
                    for ktag,ik in zip(('0003','0001'),('flair+','t1+')):
                        imgslice[ik] = np.moveaxis(imgs[ik],dim,0)[slice]
                        fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + s + '_' + ktag + '.png'
                        imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)
                    img_idx += 1
            
        a=1   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/media/jbishop/WD4/brainmets/sam_models/psam")
    parser.add_argument("--uploaddir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom_upload")
    parser.add_argument("--niftidir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom2nifti_upload")
    parser.add_argument("--datadir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/") 

    args, unknown_args = parser.parse_known_args()
    main(args.datadir)
