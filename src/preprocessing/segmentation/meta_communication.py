import sys
import os
import pysftp
import json
import shutil
from io import BytesIO

# Load env variables from .env
from dotenv import load_dotenv
load_dotenv()


#############################################

def load_confing(path="config.json"):
    with open(path, 'r') as file:
        config = json.load(file)

        return config
    
# Loading configuration

config_file = "train_config.json"

wdir = os.getcwd()
wdir = os.path.join(wdir, "src/training/segmentation") 
config = load_confing(os.path.join(wdir,config_file))

num_of_sets = config["num_of_sets"]
mode = config["mode"]
loss_function = config["loss_function"]
max_resolution = config["max_resolution"]
version = config["version"]
pretrained = config["pretrained"]

output_base_path = config["output_base_path"]

dataset = config["dataset"]

datadir = f"/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/segmentation/{mode}/data/{dataset}"



datadir_train_images = f"{datadir}/train/images"
datadir_train_interp_masks = f"{datadir}/train/interpolated_masks" if mode == "refiner" else None
datadir_train_gt_masks = f"{datadir}/train/gt_masks" if mode == "refiner" else f"{datadir}/train/masks"

datadir_val_images = f"{datadir}/val/images"
datadir_val_interp_masks = f"{datadir}/val/interpolated_masks" if mode == "refiner" else None
datadir_val_gt_masks = f"{datadir}/val/gt_masks" if mode == "refiner" else f"{datadir}/val/masks"



walltime = config["walltime"]
ncpus = config["resources"]["ncpus"]
ngpus = config["resources"]["ngpus"]
mem = config["resources"]["mem"]
gpu_mem = config["resources"]["gpu_mem"]
scratch_local = config["resources"]["scratch_local"]
gpu = config["gpu"]

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

if mode == "refiner":
    FILENAME = "refiner_train.py"
    model_num = config["model_num_refiner"]
else:
    FILENAME = "u_net_train_basic_256.py"
    model_num = config["model_num_unet"]

LOCAL_FILE_PATH = os.path.join(wdir, FILENAME)

ORIG_RUN_FILE = os.path.join(wdir, RUN_BASE)


# Get SFTP credentials from environment variables
SFTP_META_HOST = os.getenv("SFTP_META_HOST")
SFTP_META_USER = os.getenv("SFTP_META_USER")
SFTP_META_PWD = os.getenv("SFTP_META_PWD")

# Project folder on the SFTP server
AO_PROJECT_FOLDER = f"{output_base_path}/{mode}/training"

pretrained = "_pretrained_" if pretrained else ""

if mode == "refiner":
    PROJECT_USER_FOLDER = f"{AO_PROJECT_FOLDER}/refiner_m{model_num}{pretrained}{dataset}_v{version}"
else:
    PROJECT_USER_FOLDER = f"{AO_PROJECT_FOLDER}/unet_m{model_num}{pretrained}{dataset}_v{version}"



if additional_notes is not None and additional_notes != "":
    PROJECT_USER_FOLDER += f"_{additional_notes}"

OUTPUT = f"{PROJECT_USER_FOLDER}/output"

job_name = os.path.basename(PROJECT_USER_FOLDER)

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
        #if not sftp.isdir(OUTPUT):
        #    sftp.mkdir(OUTPUT)

        # Change to project folder
        sftp.chdir(PROJECT_USER_FOLDER)
        #print(sftp.pwd)

        runsh = BytesIO()
        with open(RUN_FILE, "r") as f:
            content = f.read()

        # Replace placeholders with actual values
        replacements = {
            "<FILE_NAME>": FILENAME,
            "<SETUP>": "setup.py",
            "<TYPE>": "refiner" if mode == "refiner" else "unet",
            "<CONFIG_FILE>": config_file,
            "<TRAINING_DIR>": PROJECT_USER_FOLDER,
            "<PROJECT_DIR>": "/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP",
            "<DATADIR>": datadir,
            "<DATADIR_TRAIN_IMAGES>": datadir_train_images,
            "<DATADIR_TRAIN_INTERP_MASKS>": datadir_train_interp_masks if mode == "refiner" else "",
            "<DATADIR_TRAIN_GT_MASKS>": datadir_train_gt_masks,
            "<DATADIR_VAL_IMAGES>": datadir_val_images,
            "<DATADIR_VAL_INTERP_MASKS>": datadir_val_interp_masks if mode == "refiner" else "",
            "<DATADIR_VAL_GT_MASKS>": datadir_val_gt_masks,
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
        sftp.put(os.path.join(wdir, "meta_communication.py"), f"{PROJECT_USER_FOLDER}/meta_communication.py")
        sftp.put(os.path.join(os.getcwd(), "src/setup.py"), "/storage/projects-du-praha/CVUT_Fsv_AO/Matyas_DP/setup.py")
        if mode == "refiner":
            sftp.put(os.path.join(wdir, "refiner_train.py"))
        else:
            sftp.put(os.path.join(wdir, "u_net_train_basic_256.py"))

        #print(sftp.listdir())

        print(sftp.execute(f"qsub {PROJECT_USER_FOLDER}/{os.path.basename(RUN_FILE)}"))

        os.remove(RUN_FILE)

        # end
except (pysftp.SSHException, pysftp.AuthenticationException) as e:
    print(f"SFTP connection failed: {e}")
    sys.exit(1)