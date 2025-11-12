from model import CleanUNet
import json 
import torch
import torchaudio
from scipy.io.wavfile import write as wavwrite
import resampy

MODES = {
    "full" : {
        "config" : "./config/DNS-large-full.json",
        "checkpoint" : "https://raw.githubusercontent.com/NVIDIA/CleanUNet/main/exp/DNS-large-full/checkpoint/pretrained.pkl"
    },
    "high" : {
        "config" : "./config/DNS-large-high.json",
        "checkpoint" : "https://raw.githubusercontent.com/NVIDIA/CleanUNet/main/exp/DNS-large-high/checkpoint/pretrained.pkl"
    }
}

def download_weight(mode):
    import requests

    url = MODES[mode]["checkpoint"]
    output_file = f"./{mode}.pkl"

    response = requests.get(url)
    with open(output_file, "wb") as f:
        f.write(response.content)
    return output_file


class Inference(object):
    def __init__(self, model_type="full", device="cuda"):
        model_path = download_weight(model_type)
        config_path = MODES[model_type]
        with open(config_path) as f:
            data = f.read()
        config = json.loads(data)
        self.SAMPLE_RATE = config["trainset_config"]["sample_rate"]
        network_config = config["network_config"] 
        self.net = CleanUNet(**network_config).cuda()
        checkpoint = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.device = device
        self.net.eval().to(self.device)
    
    def inference(self, audio_path, output_path):
        noisy_audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.SAMPLE_RATE:
            noisy_audio = resampy.resample(noisy_audio, sample_rate, self.SAMPLE_RATE)
        noisy_audio = noisy_audio.unsqueeze(0).to(self.device)
        generated_audio = self.net(noisy_audio)
        output_audio = generated_audio[0].squeeze().detach().cpu().numpy()
        if output_path is not None:
            wavwrite(output_path, self.SAMPLE_RATE, output_audio )
            return output_path
        else:
            return output_audio
    
