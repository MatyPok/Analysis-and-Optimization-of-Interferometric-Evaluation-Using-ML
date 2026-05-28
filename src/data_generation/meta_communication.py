import sys
import os
import pysftp
#import shutil
from io import BytesIO

import json

# Load env variables from .env
from dotenv import load_dotenv

load_dotenv()

# For run: python3 meta_communication.py interferogram_zernike_set_for_ftm.py run.sh config_0.json

# Fetch filename from cmd args
FILENAME = sys.argv[1]
RUNFILE = sys.argv[2]
config_file = sys.argv[3]

#Loading paramters from extern json config file
def load_config(path="config.json"):
    with open(path, 'r') as file:
        config = json.load(file)
        return config


# SFTP connection details from env variables
SFTP_META_HOST = os.getenv("SFTP_META_HOST")
SFTP_META_USER = os.getenv("SFTP_META_USER")
SFTP_META_PWD = os.getenv("SFTP_META_PWD")

# Project folder on the SFTP server
AO_PROJECT_FOLDER = "/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/zernikes_phases/data"
PROJECT_USER_FOLDER = f"{AO_PROJECT_FOLDER}/intens_phases_tilt_target_zernikes_464x464_10k_k1_mag_40_60_noise40db_var_Zs_A"

#AO_PROJECT_FOLDER = "/storage/projects/CVUT_Fsv_AO.old/ML_Interferometry/DATA/"
#PROJECT_USER_FOLDER = f"{AO_PROJECT_FOLDER}/data_zernikes_w_ftm_128x128_30k_k1"

# Create connection
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

with pysftp.Connection(host=SFTP_META_HOST, username=SFTP_META_USER, password=SFTP_META_PWD, cnopts=cnopts) as sftp:

    # Change directory to the group folder
    sftp.chdir(AO_PROJECT_FOLDER)
    #print(sftp.listdir(AO_PROJECT_FOLDER))

    # Create project folder if it does not exist
    if not sftp.isdir(PROJECT_USER_FOLDER):
        sftp.mkdir(PROJECT_USER_FOLDER)

    # Change to project folder
    sftp.chdir(PROJECT_USER_FOLDER)
    #print(sftp.pwd)

    runsh = BytesIO()
    with open(RUNFILE, "r") as f:
        [
            runsh.write(
                str.encode(line.replace("<FILE_NAME>", FILENAME))
            )
            for line in f.readlines()
        ]

    with open(RUNFILE, "wb") as f:
        f.write(runsh.getbuffer())

    # Upload file to meta
    sftp.put(f"{RUNFILE}")
    sftp.put(f"{FILENAME}")
    sftp.put(f"{config_file}")
    sftp.put("polynomials.py")
    sftp.put("coefficients.py")
    sftp.put("noise.py")
    #print(sftp.listdir())

    #os.remove("run.sh")

    print(sftp.execute(f"qsub {PROJECT_USER_FOLDER}/{RUNFILE}"))

    # end
