from clean_unet.modules import CleanUNet
import json 
import torch
import torchaudio
from scipy.io.wavfile import write as wavwrite
import resampy

MODES = {
    "full" : {
        "config" : "./clean_unet/config/DNS-large-full.json",
        "checkpoint" : "./clean_unet/full.pkl"
    },
    "high" : {
        "config" : "./clean_unet/config/DNS-large-high.json",
        "checkpoint" : "./clean_unet/high.pkl"
    }
}



class InferenceCleanUNet(object):
    def __init__(self, model_type="full", device="cuda"):
        model_path = MODES[model_type]["checkpoint"]
        config_path = MODES[model_type]["config"]
        with open(config_path, "r") as f:
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
        torch.cuda.empty_cache()
        if output_path is not None:
            wavwrite(output_path, self.SAMPLE_RATE, output_audio )
            return output_path
        else:
            return output_audio
    
