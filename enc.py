import numpy as np
import matplotlib.pyplot as plt

def generate_waveform(letter):
    if letter == 'J':
        return np.sin(np.linspace(0, 2*np.pi, 100))
    elif letter == 'E':
        return np.exp(np.linspace(-2, 2, 100))
    elif letter == 'F':
        return np.polyval([1, 2, 1], np.linspace(-1, 1, 100))
    else:
        raise ValueError("Invalid letter")

def add_parity_check(waveform):
    parity_bit = np.sum(waveform) % 2
    return np.append(waveform, parity_bit)

def decode_with_parity_check(waveform_with_parity):
    parity_bit = waveform_with_parity[-1]
    waveform = waveform_with_parity[:-1]
    calculated_parity = np.sum(waveform) % 2
    if calculated_parity != parity_bit:
        # Handle error (e.g., request retransmission)
        print("Error detected!")
    else:
        return waveform

# Example usage:
waveform_j = generate_waveform('J')
waveform_j_with_parity = add_parity_check(waveform_j)

# ... similar for 'E' and 'F'

# Transmission and reception (simulated)
received_waveform_j = waveform_j_with_parity  # Simulate error-free transmission

# Decoding
decoded_waveform_j = decode_with_parity_check(received_waveform_j)

# Plot the waveforms
plt.plot(waveform_j, label='Original J')
plt.plot(decoded_waveform_j, label='Decoded J')
plt.legend()
plt.show()
