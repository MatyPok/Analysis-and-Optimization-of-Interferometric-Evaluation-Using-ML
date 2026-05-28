import sys
import os
import pysftp
import json
import shutil
from io import BytesIO

# Load env variables from .env
from dotenv import load_dotenv
load_dotenv()


if len (sys.argv) == 1:
    FILENAME = "training_zernike_phase.py"
    config_file = "train_config.json"
elif len(sys.argv) == 2:
    FILENAME = sys.argv[1]
    config_file = "train_config.json"
else:
    FILENAME = sys.argv[1]
    config_file = sys.argv[2]


#############################################

def load_confing(path="config.json"):
    with open(path, 'r') as file:
        config = json.load(file)

        return config
    
# Loading configuration

wdir = os.getcwd()
wdir = os.path.join(wdir, "src/training/zernikes_phases_w_ftm") 
config = load_confing(os.path.join(wdir,config_file))

compute_with_noise = config["compute_with_noise"]
num_of_sets = config["num_of_sets"]
model = config["model"]
loss_function = config["loss_function"]
weight_loss = config["weight_loss"]
pretrained = config["pretrained"]
fine_tuning = config["fine_tuning"]
resolution = config["resolution"]
freeze_backbone = config["freeze_backbone"]
correlation_loss = config["correlation_loss"]

data_base_path = config["data_base_path"]
output_base_path = config["output_base_path"]

datadir = config["datadir"]

datadir_intensity = datadir + "/output/intensity_circular.npz"
datadir_phase = datadir + "/output/phases_circular.npz"
datadir_labels = datadir + "/output/target_coefficients.npz"
datadir_carriers = datadir + "/output/tilt.npz"
datadir_fft = datadir + "/output/fft_log_norm.npz"
datadir_fft_no_dc = datadir + "/output/fft_log_norm_minus_dc.npz"

models = "models.py"
interferogram_generation = "interferogram.py"

walltime = config["walltime"]
ncpus = config["resources"]["ncpus"]
ngpus = config["resources"]["ngpus"]
mem = config["resources"]["mem"]
gpu_mem = config["resources"]["gpu_mem"]
scratch_local = config["resources"]["scratch_local"]
gpu = config["gpu"]

normalize_y = config["normalize_y"]
normalize_X = config["normalize_X"]

use_fft = config["use_fft"]
use_fft_no_dc = config["use_fft_no_dc"]

additional_notes = config["additional_notes"]

if gpu:
    resources = f"select=1:ncpus={ncpus}:ngpus={ngpus}:mem={mem}:gpu_mem={gpu_mem}:scratch_local={scratch_local}"
    if walltime >= "24:00:00":
        server = "gpu_long@pbs-m1.metacentrum.cz"
    else:
        server = "gpu@pbs-m1.metacentrum.cz"
else:
    server = "default@pbs-m1.metacentrum.cz"
    resources = f"select=1:ncpus={ncpus}:mem={mem}:scratch_local={scratch_local}"

###############################

RUN_BASE = "run_train.sh"

LOCAL_FILE_PATH = os.path.join(wdir, FILENAME)

ORIG_RUN_FILE = os.path.join(wdir, RUN_BASE)


# Get SFTP credentials from environment variables
SFTP_META_HOST = os.getenv("SFTP_META_HOST")
SFTP_META_USER = os.getenv("SFTP_META_USER")
SFTP_META_PWD = os.getenv("SFTP_META_PWD")

# Project folder on the SFTP server
AO_PROJECT_FOLDER = output_base_path

PROJECT_USER_FOLDER = f"{AO_PROJECT_FOLDER}/{model}_{int(num_of_sets/1000)}k_{resolution}_zernike"

PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_pretrained" if pretrained else PROJECT_USER_FOLDER
PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_finetuned" if fine_tuning else PROJECT_USER_FOLDER

PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_normXy" if normalize_y and normalize_X else PROJECT_USER_FOLDER
PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_normy" if normalize_y and not normalize_X else PROJECT_USER_FOLDER
PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_normX" if normalize_X and not normalize_y else PROJECT_USER_FOLDER

PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_corrloss" if correlation_loss else PROJECT_USER_FOLDER
PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + f"_weighted_{loss_function}" if weight_loss else PROJECT_USER_FOLDER + f"_{loss_function}"

PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_fft" if use_fft else PROJECT_USER_FOLDER
PROJECT_USER_FOLDER = PROJECT_USER_FOLDER + "_fft_no_dc" if use_fft_no_dc else PROJECT_USER_FOLDER





if additional_notes is not None and additional_notes != "":
    PROJECT_USER_FOLDER += f"_{additional_notes}"

job_name = os.path.basename(PROJECT_USER_FOLDER)

OUTPUT = f"{PROJECT_USER_FOLDER}/output"


# Create connection
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None


# Create copy od run_train.sh for this training with repclaced parameters
RUN_FILE = os.path.join(wdir, "run_train_copy.sh")
shutil.copy(ORIG_RUN_FILE, RUN_FILE)


try:
    with pysftp.Connection(host=SFTP_META_HOST, username=SFTP_META_USER, password=SFTP_META_PWD, cnopts=cnopts) as sftp:

        # Change directory to the group folder
        sftp.chdir(AO_PROJECT_FOLDER)
        #print(sftp.listdir(AO_PROJECT_FOLDER))

        # Create project folder if it does not exist
        if not sftp.isdir(PROJECT_USER_FOLDER):
            sftp.mkdir(PROJECT_USER_FOLDER)

        # Create output folder if it does not exist
        if not sftp.isdir(OUTPUT):
            sftp.mkdir(OUTPUT)

        # Change to project folder
        sftp.chdir(PROJECT_USER_FOLDER)
        #print(sftp.pwd)

        runsh = BytesIO()
        with open(RUN_FILE, "r") as f:
            content = f.read()

        # Replace placeholders with actual values
        replacements = {
            "<FILE_NAME>": FILENAME,
            "<TYPE>": "zernike_ftm",
            "<CONFIG_FILE>": config_file,
            "<DATADIR>": PROJECT_USER_FOLDER,
            "<DATADIR_INTENSITY>": datadir_intensity,
            "<DATADIR_PHASE>": datadir_phase,
            "<DATADIR_LABELS>": datadir_labels,
            "<DATADIR_CARRIERS>":datadir_carriers,
            "<DATADIR_FFT>":datadir_fft,
            "<DATADIR_FFT_NO_DC>":datadir_fft_no_dc ,
            "<MODELS>": models,
            "<INTERFEROGRAM_GENERATION>": interferogram_generation,
            "<WALLTIME>": walltime,
            "<JOB_NAME>": job_name,
            "<RESOURCES>": resources,
            "<SERVER>": server
        }

        for placeholder, value in replacements.items():
            content = content.replace(placeholder, str(value))

        runsh.write(content.encode())

        with open(RUN_FILE, "wb") as f:
            f.write(runsh.getbuffer())

        # Upload file to meta
        sftp.put(RUN_FILE, f"{PROJECT_USER_FOLDER}/{os.path.basename(RUN_FILE)}")
        sftp.put(LOCAL_FILE_PATH, f"{PROJECT_USER_FOLDER}/{FILENAME}")
        sftp.put(os.path.join(wdir, config_file), f"{PROJECT_USER_FOLDER}/{config_file}")
        sftp.put(os.path.join(wdir, "models.py"))
        sftp.put(os.path.join(wdir, "interferogram.py"))
        sftp.put(os.path.join(wdir, "polynomials.py"))

        #print(sftp.listdir())

        print(sftp.execute(f"qsub {PROJECT_USER_FOLDER}/{os.path.basename(RUN_FILE)}"))

        os.remove(RUN_FILE)

        # end
except (pysftp.SSHException, pysftp.AuthenticationException) as e:
    print(f"SFTP connection failed: {e}")
    sys.exit(1)