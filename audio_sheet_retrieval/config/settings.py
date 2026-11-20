import os

# Paths are configured through environment variables, e.g.
#   export EXP_ROOT=/path/to/experiments
#   export DATA_ROOT_MSMD=/path/to/msmd_aug_v1-1_no-audio
#
# EXP_ROOT:       where model folders are created and parameters/results are dumped
# DATA_ROOT_MSMD: root directory of the MSMD dataset
EXP_ROOT = os.environ.get("EXP_ROOT", os.path.join(os.getcwd(), "experiments"))
DATA_ROOT_MSMD = os.environ.get("DATA_ROOT_MSMD", os.path.join(os.getcwd(), "data", "msmd"))
