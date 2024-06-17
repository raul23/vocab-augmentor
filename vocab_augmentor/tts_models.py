#####################
# Classes: TTS models
#####################
class FacebookTTS:
    def __init__(self, model_name, verbose=False):
        """
        Initializes the FacebookTTS class with the specified model name.

        Args:
            model_name (str): The name of the pretrained model.
            verbose (bool, optional): Whether to print verbose outputs. Default is False.
        
        Attributes:
            torch: The imported torch module.
            scipy: The imported scipy module.
            np: The imported numpy module.
            model (VitsModel): The loaded VitsModel from transformers.
            tokenizer (AutoTokenizer): The loaded AutoTokenizer from transformers.
        """
        self.model_name = model_name
        
        ## Import necessary modules
        # from transformers import VitsModel, AutoTokenizer
        VitsModel = import_module("transformers", "VitsModel")
        AutoTokenizer = import_module("transformers", "AutoTokenizer")
        # import torch
        self.torch = import_module("torch")
        # import scipy
        self.scipy = import_module("scipy")
        # import numpy
        self.np = import_module("numpy", alias="np")
        
        # Load model and tokenizer
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_speech(self, text, **kwargs):
        """
        Generates speech from text input using the loaded model.

        Args:
            text (str): The input text to generate speech from.
            **kwargs: Additional keyword arguments for tokenization.
        
        Returns:
            str: The path to the generated audio file.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Generate waveform from model
        with self.torch.no_grad():
            output = self.model(**inputs).waveform
            
        # Generate a random temporary filename with .wav suffix
        audio_path = get_random_temp_filename(".wav")
        
        # Convert PyTorch tensor to NumPy array
        output_np = output.cpu().numpy()
        
        # Ensure the data is in the correct shape (1D array)
        if len(output_np.shape) > 1:
            output_np = output_np.squeeze()
        
        # Write the NumPy array to a WAV file
        self.scipy.io.wavfile.write(audio_path, rate=self.model.config.sampling_rate, data=output_np)
        
        return audio_path
       
    
class MeloTTS:
    def __init__(self, verbose=False):
        """
        Initializes the MeloTTS class with the specified verbosity setting.

        Args:
            verbose (bool, optional): Whether to print verbose outputs. Default is False.
        
        Attributes:
            client (Client): The Gradio Client instance for interacting with MeloTTS.
        """
        self.model_name = "MeloTTS"
        
        ## Import Gradio Client module
        # from gradio_client import Client
        Client = import_module("gradio_client", "Client")
        
        # Initialize Gradio Client for MeloTTS
        self.client = Client("mrfakename/MeloTTS", verbose=verbose)
    
    def generate_speech(self, text, lang, event_listener="predict"):
        """
        Generates speech from text input using the MeloTTS model.

        Args:
            text (str): The input text to generate speech from.
            lang (str): The language code for selecting the speaker.
            event_listener (str, optional): The type of event listener ('predict' or 'submit'). Default is 'predict'.
        
        Returns:
            str or Job: The path to the generated audio file if event_listener is 'predict',
                       or a job if event_listener is 'submit'.
        
        Raises:
            ValueError: If an unsupported event listener is provided.
        """
        # Determine speaker based on language
        if lang == "en":
            speaker = "EN-Default"
        else:
            speaker = lang.upper()
            
        # Perform prediction or submit job based on event_listener
        if event_listener == "predict":
            audio_path = self.client.predict(
                text=text,
                speaker=speaker,
                speed=1,
                language=lang.upper(),
                api_name="/synthesize"
            )
            return audio_path
        elif event_listener == "submit":
            job = self.client.submit(
                text=text,
                speaker=speaker,
                speed=1,
                language=lang.upper(),
                api_name="/synthesize"
            )
            return job
        else:
            raise ValueError(f"Unsupported event listener: {event_listener}")
