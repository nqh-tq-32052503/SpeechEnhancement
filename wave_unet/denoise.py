
import torch 
from wave_unet.modules.modules import UNet, UNet_ResNet
from wave_unet.modules.config import N_FFT, HOP_LENGTH_FFT, HOP_LENGTH_FRAME, SAMPLE_RATE, FRAME_LENGTH, REDUCE_RATE, MIN_DURATION
from wave_unet.processor import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio, inv_scaled_ou, scaled_in
import soundfile as sf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InferenceWaveUNet(object):
    def __init__(self, model_type):
        if model_type == 'Unet':
            model = UNet(start_fm=32)
            model.load_state_dict(torch.load('./model/unet.pth', map_location='cpu'))
        else:
            model = UNet_ResNet(start_fm=16)
            model.load_state_dict(torch.load('./model/unetres.pth', map_location='cpu'))
        self.model = model 
        self.model.eval().to(device)
    
    def inference(self, audio_path, output_path):
        audio = audio_files_to_numpy(audio_path, SAMPLE_RATE,
                                FRAME_LENGTH, HOP_LENGTH_FRAME, MIN_DURATION)
        dim_square_spec = int(N_FFT / 2) + 1
        m_amp_db,  m_pha = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, N_FFT, HOP_LENGTH_FFT)
        X_in = torch.from_numpy(scaled_in(m_amp_db)).unsqueeze(1).to(device, dtype=torch.float)
        with torch.no_grad():
            X_pred = self.model(X_in)
            pred_amp_db = inv_scaled_ou(X_pred.squeeze().detach().cpu().numpy(), REDUCE_RATE)
            X_denoise = m_amp_db - pred_amp_db
            ou_audio = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha, FRAME_LENGTH, HOP_LENGTH_FFT)
            nb_samples = ou_audio.shape[0]
            denoise_long = ou_audio.reshape(1, nb_samples*FRAME_LENGTH)*10
            sf.write(output_path, denoise_long[0, :], SAMPLE_RATE)
