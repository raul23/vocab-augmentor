import logging
# Suppress the specific warning by setting the logging level to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import transformers

from huggingface_hub import notebook_login

if __name__ == "__main__":
    # Login to Hugging Face
    notebook_login()
