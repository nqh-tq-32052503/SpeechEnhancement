from model import DCUnet20
from processor import SpeechProcessing
import torch
import torchaudio
import resampy

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 
DEVICE = "cuda"

MODES = {
    "noise2clean" : "./weights/noise2clean.pth",
    "noise2noise" : "./weights/noise2noise.pth"
}

class Inference(object):
    def __init__(self, model_type):
        model_weights_path = MODES[model_type]
        self.model = DCUnet20(N_FFT, HOP_LENGTH)
        checkpoint = torch.load(model_weights_path,
                                map_location=torch.device('cpu')
                                )
        self.model.load_state_dict(checkpoint)
        self.model.eval().to(DEVICE)
        self.processor = SpeechProcessing(N_FFT, HOP_LENGTH)
    
    def inference(self, audio_path, output_path):
        _, sr = torchaudio.load(audio_path)
        with torch.no_grad():
            input_tensors = self.processor(audio_path)
            output_tensors = []
            for input_tensor in input_tensors:
                input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
                output_tensor = self.model(input_tensor, is_istft=True)
                output_tensors.append(output_tensor)
            output = torch.cat(output_tensors, dim=1)
            self.processor.save(output, output_path, sample_rate=sr)