import torch
import scipy.signal
import numpy as np
from typing import List
from fft_conv_pytorch import fft_conv
import librosa


def load_audio(path, target_sr: int = 48000, mono=False, offset=0.0, duration=None) -> np.ndarray:
    """
    return y : shape=(n_channels, n_samples)
    """
    # mono: 是否是单通道   offset: 在此时间后开始阅读   duration: 获取音频的时长  后两个单位都是秒
    y, orig_sr = librosa.load(path, sr=None, mono=mono, offset=offset, duration=duration)

    if target_sr:
        y = resample(y, orig_sr=orig_sr, target_sr=target_sr)

    return np.atleast_2d(y)


def resample(signal: np.ndarray, orig_sr: int, target_sr: int, **kwargs) -> np.ndarray:
    """signal: (N,) or (num_channel, N)"""
    return librosa.resample(y=signal, orig_sr=orig_sr, target_sr=target_sr, res_type="polyphase", **kwargs)


def crop_rir(rir, target_length):
    """
    裁剪RIR到指定的长度。如果长度不够则补0，否则裁掉多余的部分。
    :param rir:
    :param target_length:
    :return:
    """
    n_channels, num_samples = rir.shape

    # by default: all the test rirs will be aligned such that direct impulse starts with 90 sample delay@48kHz
    if num_samples < target_length:
        out_rir = np.zeros((n_channels, target_length))
        out_rir[:, :num_samples] = rir
    else:
        out_rir = rir[:, :target_length]

    return out_rir


def get_octave_filters():
    """10 octave bandpass filters, each with order 1023
    Return
        firs : shape = (10, 1, 1023)
    """
    f_bounds = []
    f_bounds.append([22.3, 44.5])
    f_bounds.append([44.5, 88.4])
    f_bounds.append([88.4, 176.8])
    f_bounds.append([176.8, 353.6])
    f_bounds.append([353.6, 707.1])
    f_bounds.append([707.1, 1414.2])
    f_bounds.append([1414.2, 2828.4])
    f_bounds.append([2828.4, 5656.8])
    f_bounds.append([5656.8, 11313.6])
    f_bounds.append([11313.6, 22627.2])

    firs: List = []
    for low, high in f_bounds:
        fir = scipy.signal.firwin(
            1023,
            np.array([low, high]),
            pass_zero='bandpass',
            window='hamming',
            fs=48000,
        )
        firs.append(fir)

    firs = np.array(firs)
    firs = np.expand_dims(firs, 1)
    return firs


def batch_convolution(signal, filter):
    """Performs batch convolution with pytorch fft convolution.
    Args
        signal : torch.FloatTensor (batch, n_channels, num_signal_samples)
        filter : torch.FloatTensor (batch, n_channels, num_filter_samples)
    Return
        filtered_signal : torch.FloatTensor (batch, n_channels, num_signal_samples)
    """
    batch_size, n_channels, signal_length = signal.size()
    _, _, filter_length = filter.size()

    # Pad signal in the beginning by the filter size
    padded_signal = torch.nn.functional.pad(signal, (filter_length, 0), 'constant', 0)

    # Transpose : move batch to channel dim for group convolution
    padded_signal = padded_signal.transpose(0, 1)

    filtered_signal = fft_conv(padded_signal.double(), filter.double(), padding=0, groups=batch_size).transpose(0, 1)[
                      :, :, :signal_length
                      ]

    filtered_signal = filtered_signal.type(signal.dtype)

    return filtered_signal


def add_noise_batch(batch_signal, noise, snr_db):
    """Add noise to signal with the given SNR
    Args
        batch_signal : torch.FloatTensor. shape=(batch, 1, signal_length)
        noise : torch.FloatTensor. shape=(batch, 1, signal_length)
        snr_db : torch.FloatTensor. shape=(batch, 1)
    Return
        noise_added_signal : torch.FloatTensor. shape=(batch, 1, signal_length)
    """
    b, n, l = batch_signal.size()

    mean_square_signal = torch.mean(batch_signal ** 2, dim=2)
    signal_level_db = 10 * torch.log10(mean_square_signal)
    noise_db = signal_level_db - snr_db
    mean_square_noise = torch.sqrt(10 ** (noise_db / 10))
    mean_square_noise = torch.unsqueeze(mean_square_noise, dim=2)
    mean_square_noise = mean_square_noise.repeat(1, 1, l)
    modified_noise = torch.mul(noise, mean_square_noise)

    return batch_signal + modified_noise


def rms_normalize(sig: np.ndarray, rms_level=0.1):
    """
    sig : shape=(channel, signal_length)
    rms_level : linear gain value
    """
    # linear rms level and scaling factor
    # r = 10 ** (rms_level / 10.0)
    a = np.sqrt((sig.shape[1] * rms_level ** 2) / (np.sum(sig ** 2) + 1e-7))

    # normalize
    y = sig * a
    return y


def rms_normalize_batch(sig: torch.Tensor, rms_level=0.1):
    """
    sig : shape=(batch, channel, signal_length)
    Returns a tuple of the normalized signals and the scaling factors used for normalization.
    返回归一化后的batch以及反归一化系数
    """
    # Calculate scaling factor to achieve target RMS level
    original_rms = torch.sqrt(torch.mean(sig ** 2, dim=2, keepdims=True) + 1e-7)
    target_rms = rms_level * torch.ones_like(original_rms)
    scaling_factors = torch.where(original_rms > 0, target_rms / original_rms, torch.ones_like(original_rms))

    # Normalize
    y = sig * scaling_factors

    return y, scaling_factors


def rms_denormalize_batch(normalized_sig: torch.Tensor, scaling_factors: torch.Tensor):
    """
    反归一化处理后的音频信号。

    参数:
    normalized_sig : shape=(batch, channel, signal_length)，已归一化的音频信号张量。
    scaling_factors : shape=(batch, channel, 1)，归一化时使用的缩放因子张量。

    返回:
    原始音频信号张量，shape=(batch, channel, signal_length)。
    """
    # 检查输入张量的形状是否匹配
    if normalized_sig.shape[:2] != scaling_factors.shape[:2]:
        raise ValueError("normalized_sig 和 scaling_factors 的 batch 和 channel 维度必须匹配")

    # 确保scaling_factors是正确的形状，即(batch, channel, 1)
    if len(scaling_factors.shape) != 3 or scaling_factors.shape[2] != 1:
        raise ValueError("scaling_factors 必须具有形状 (batch, channel, 1)")

    # 反归一化：将归一化后的信号除以归一化时的缩放因子
    original_sig = normalized_sig / scaling_factors

    return original_sig


def peak_normalize(sig: np.ndarray, peak_val):
    peak = np.max(np.abs(sig[:, :512]), axis=-1, keepdims=True)
    sig = np.divide(sig, peak + 1e-7)
    sig = sig * peak_val
    return sig


def peak_normalize_batch(sig: torch.Tensor, peak_val=1.0):
    """
    sig : shape=(batch, channel, signal_length)
    peak_val : 目标峰值幅度，默认为 1.0
    返回已归一化的音频信号和用于归一化的缩放因子。
    """
    # 计算每个信号的最大绝对值作为缩放因子
    scaling_factors = torch.max(torch.abs(sig), dim=2, keepdim=True).values

    # 避免除以零的情况
    scaling_factors = torch.where(scaling_factors == 0, torch.ones_like(scaling_factors), scaling_factors)

    # 归一化：将信号除以其最大绝对值并乘以目标峰值幅度
    normalized_sig = torch.div(sig, scaling_factors) * peak_val

    return normalized_sig, scaling_factors


def peak_denormalize_batch(normalized_sig: torch.Tensor, scaling_factors: torch.Tensor, peak_val=1.0):
    """
    反归一化处理后的音频信号。

    参数:
    normalized_sig : shape=(batch, channel, signal_length)，已归一化的音频信号张量。
    scaling_factors : shape=(batch, channel, 1)，归一化时使用的缩放因子张量。
    peak_val : 归一化时使用的目标峰值幅度，默认为 1.0

    返回:
    原始音频信号张量，shape=(batch, channel, signal_length)。
    """
    # 检查输入张量的形状是否匹配
    if normalized_sig.shape[:2] != scaling_factors.shape[:2]:
        raise ValueError("normalized_sig 和 scaling_factors 的 batch 和 channel 维度必须匹配")

    # 确保scaling_factors是正确的形状，即(batch, channel, 1)
    if len(scaling_factors.shape) != 3 or scaling_factors.shape[2] != 1:
        raise ValueError("scaling_factors 必须具有形状 (batch, channel, 1)")

    # 反归一化：将归一化后的信号除以目标峰值幅度再乘以归一化时的缩放因子
    original_sig = normalized_sig / peak_val * scaling_factors

    return original_sig


def audio_normalize_batch(sig, type, rms_level=0.1, peak_val=1.0):
    """

    :param sig:
    :param type:
    :param rms_level:
    :param peak_val:
    :return: 两个变量，第一个是归一化后的音频，第二个是反归一化系数
    """
    if type == "peak":
        return peak_normalize_batch(sig, peak_val)
    else:
        return rms_normalize_batch(sig, rms_level)
