import subprocess
import os

# clone the community-events repository
subprocess.run(["git", "clone", "https://github.com/huggingface/community-events.git"])

# install the required packages
os.chdir("community-events/whisper-fine-tuning-event")
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# configure git credential helper
subprocess.run(["git", "config", "--global", "credential.helper", "store"])

# login to huggingface cli
subprocess.run(["huggingface-cli", "login"])

# install 🤗 libraries
subprocess.run(["pip", "install", "datasets", "git+https://github.com/huggingface/transformers",\
    "evaluate", "huggingface_hub", "jiwer", "bitsandbytes", "accelerate", "protobuf==3.19.0"])

# upgrade pip, if necessary
subprocess.run(["pip", "install", "--upgrade", "pip"])