
import matplotlib.pyplot as plt
from simulation.generate_signal import generate_signal

signal = generate_signal(1)

plt.plot(signal)
plt.title("Simulated Sensor Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
