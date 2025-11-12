import torch 
import torchaudio
import numpy as np

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 
DEVICE = "cuda"

class SpeechProcessing: 
    def __init__(self, n_fft=64, hop_length=16):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = 165000
    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform
        
    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]
        num_chunks = int(current_len / self.max_len) + 1
        outputs = []
        for index in range(num_chunks):
            output = np.zeros((1, self.max_len), dtype='float32')
            s = index * self.max_len
            e = (index + 1) * self.max_len
            output[0, -current_len:] = waveform[0, s: e]
            output = torch.from_numpy(output).to(DEVICE)
            output = torch.stft(input=output, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, normalized=True, return_complex=True)
            output = torch.view_as_real(output)
            outputs.append(output)
            current_len -= self.max_len
        return outputs
        
    def __call__(self, audio_path):
        x_noisy = self.load_sample(audio_path)
        x_noisies = self._prepare_sample(x_noisy)
        # x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, 
        #                           hop_length=self.hop_length, normalized=True, return_complex=True).to(DEVICE)
        # x_noisy_stft = torch.cat(x_noisies).to(DEVICE)
        return x_noisies
        
    def save(self, output_tensor, file_path, sample_rate=SAMPLE_RATE, bit_precision=16):
        np_output = output_tensor[0].view(-1).detach().cpu().numpy()
        np_array = np.reshape(np_output, (1,-1))
        torch_tensor = torch.from_numpy(np_array)
        torchaudio.save(file_path, torch_tensor, sample_rate, bits_per_sample=bit_precision)