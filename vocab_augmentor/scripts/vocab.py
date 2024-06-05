import logging
import os
import warnings

# Suppress the specific warning by setting the logging level to ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import transformers

from huggingface_hub import notebook_login

api_token = os.environ.get('GEMINI_API_TOKEN')

if __name__ == "__main__":
    # Login to Hugging Face
    notebook_login()
