import numpy as np
import torch
import matplotlib.pyplot as plt
import os

filename = 'info500000.txt'

def read_txt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        f.close()

    lines = lines[::2]
    return lines

if __name__ == '__main__':
    lines = read_txt(filename=filename)
    transitions = []
    for line in lines:
        line = line.strip()
        transitions.append(int(line))
        
    x = np.linspace(0, len(transitions), len(transitions), endpoint=False)
    y = np.array(transitions)
    plt.plot(x, y)
    plt.show()
    