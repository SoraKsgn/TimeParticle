import torch
import torch.nn as nn


def FFT_harmonic(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    frequency_list[1] = 0
    frequencies = torch.arange(len(frequency_list))

    selected_frequencies = []
    periods = []

    for _ in range(k):
        max_idx = torch.argmax(frequency_list)
        selected_frequencies.append(frequencies[max_idx].item())
        periods.append(x.shape[1] // frequencies[max_idx].item())

        to_remove = torch.tensor([max_idx, 2 * max_idx, max_idx // 2])
        to_remove = to_remove[to_remove < len(frequency_list)]
        frequency_list[to_remove] = 0

    return periods

def FFT_basic(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    frequency_list[1] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.linear = nn.Linear(self.seq_len, self.pred_len)



    def forecast(self, x_enc):
        # period = FFT_basic(x_enc, self.k)
        period = FFT_harmonic(x_enc, self.k)
        print('Period/Patch_len', period)
        x_enc = self.linear(x_enc.permute(0, 2, 1))
        # Encoder
        return x_enc.permute(0, 2, 1)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)

        return dec_out
