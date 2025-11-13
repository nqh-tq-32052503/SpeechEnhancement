from clean_unet.denoise import InferenceCleanUNet
from dc_unet.denoise import InferenceDcUNet
from wave_unet.denoise import InferenceWaveUNet

LIST_MODELS = ["clean.full", "clean.high", "dc.n2c", "dc.n2n", "wave.unet", "wave.resnet"]
DEVICE = "cuda"
print("Supported models: ", LIST_MODELS)
class DenoisingExecutive(object):
    def __init__(self, model_name):
        assert model_name in LIST_MODELS, model_name + " has not been supported"
        self.model_name = model_name
        self.init_model()

    
    def init_model(self):
        model_type, variant = self.model_name.split(".")
        if model_type == "clean":
            self.model = InferenceCleanUNet(model_type=variant, device=DEVICE)
        elif model_type == "dc":
            if variant == "n2c":
                self.model = InferenceDcUNet(model_type="noise2clean")
            elif variant == "n2n":
                self.model = InferenceDcUNet(model_type="noise2noise")
            else:
                pass
        elif model_type == "wave":
            if variant == "unet":
                self.model = InferenceWaveUNet(model_type="Unet")
            elif variant == "resnet":
                self.model = InferenceWaveUNet(model_type="Unet-Resnet")
            else:
                pass 
    
    def inference(self, audio_path, output_path):
        self.model.inference(audio_path, output_path)

