


# tests

model = MNv2(width_mult=1.)
model.summary()


import time
times = []
attempts = 300
last = 10
items = 1

import numpy as np
dummy = np.zeros((items,224,224,3), dtype='float')
for i in range(attempts):
    a = time.time()
    model.model.predict(dummy)
    b = time.time()
    times.append(b-a)

print(f"AVG:{np.mean(times)/items:0.4f}s")
print(f"AVG(last):{np.mean(times[-last:])/items:0.4f}s")
