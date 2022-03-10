import torch
import numpy as np


def ShowCoutour(activations, cidx, n = 100):
    colors = ['green', 'dodgerblue', 'orange']
    x = torch.range(-n / 2, n / 2 - 1).numpy()
    x, y = np.meshgrid(x, x)
    for j, z in enumerate(activations.t()[:10]):
        z = z.view(n, n).cpu().detach().numpy()
        plt.contour(x, y, z, [0], colors=colors[cidx], linewidths=0.1)
    return


def _count_batch_transition_torch(array: torch.Tensor) -> torch.Tensor:
    array_prev = array[:-1]
    array_next = array[1:]
    mismatches = torch.not_equal(array_prev, array_next)
    transitions = torch.sum(mismatches, dim=-1, keepdim=True) #对除了第一维度之外的维度做逻辑或
    return transitions

def EncodingReLU(f):
    fEncoding = torch.gt(f, torch.zeros_like(f))
    return fEncoding


def LinearSplitEncoding(features, type = 'ReLU'):
    '''
    f1, f2, f3 = features
    if type == 'ReLU':
        Encoding = EncodingReLU
    else:
        Encoding = EncodingHardtanh

    f1Encoding = Encoding(f1)
    f2Encoding = Encoding(f2)
    f3Encoding = Encoding(f3)

    return [f1Encoding, f2Encoding, f3Encoding]
    '''
    f1 = features
    if type == 'ReLU':
        Encoding = EncodingReLU
    else:
        Encoding = EncodingHardtanh

    f1Encoding = Encoding(f1)

    return f1Encoding

def EncodingHardtanh(f):
    n, d = f.shape
    f = f.view(-1)
    positiveIndices = torch.nonzero(torch.gt(f, torch.ones_like(f)))
    f[positiveIndices] = 1
    negativeIndices = torch.nonzero(torch.lt(f, -1 * torch.ones_like(f)))
    f[negativeIndices] = -1
    zeroIndices = torch.nonzero((f > -1) & (f < 1))
    f[zeroIndices] = 0
    f = f.view(n, d)
    return f
def count_continuous_transition(patterns):
    num_samples = patterns.shape[0]
