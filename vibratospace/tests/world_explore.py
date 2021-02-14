import matplotlib.pyplot as plt
import os
from vibratospace.src.python.defaults import DATA_PATH
from vibratospace.src.python.util import load_data

data = load_data(
    os.path.join(DATA_PATH, 'data.pickle')
)

datum = data[1]

sp = datum['world']['sp']
ap = datum['world']['ap']
f0 = datum['world']['f0']

plt.imshow(sp.T, origin='lower', aspect='auto')
plt.show()

plt.imshow(ap.T, origin='lower', aspect='auto')
plt.show()