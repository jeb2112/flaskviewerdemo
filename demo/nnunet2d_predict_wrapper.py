# script wraps the nnUNetv2_predict command to provide some arguments for
# local path on the back-end and stream stdout

import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import nibabel as nb
import shutil
import subprocess
import sys

def main(datadir,env,dataset,model):
    inputdir = os.path.join(datadir,'nnUNet_raw','flask','imagesTs')
    outputdir = os.path.join(datadir,'nnUNet_predictions','flask')

    try:
        shutil.rmtree(outputdir)
    except FileNotFoundError:
        pass
    os.makedirs(outputdir,exist_ok=True)
    
    
    # Use os.system which will show output directly in terminal
    # Set PYTHONUNBUFFERED to ensure no buffering
    os.environ['PYTHONUNBUFFERED'] = '1'
    if False:
        cmd = f"conda run -n {env} nnUNetv2_predict -i {inputdir} -o {outputdir} -d {dataset} -c {model}"
        return_code = os.system(cmd)
        if return_code != 0:
            print(f"Process exited with code {return_code}", file=sys.stderr, flush=True)
    else:
        cmd = ["nnUNetv2_predict", "-i", inputdir, "-o", outputdir, "-d", dataset, "-c", model]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        # Stream output line by line
        for line in iter(process.stdout.readline, ""):
            print(line, end='', flush=True)  # Ensures real-time output in terminal

        process.stdout.close()
        process.wait()
    
    
    print(f"Process completed for model {model}.", file=sys.stderr, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/") 
    parser.add_argument("--env", type=str, default="flask")
    parser.add_argument("--dataset", type=str, default="139") # ie whatever nnUNet dataset # has been used to identify the model
    parser.add_argument("--model", type=str,default='2d')
    args = parser.parse_args()
    main(args.datadir,args.env,args.dataset,args.model)
