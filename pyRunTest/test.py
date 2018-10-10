import numpy as np
from array import array
from testlssp import testlssp
a = np.ndarray((10,), dtype=np.int32)
l = testlssp()
# print(l.main(2, np.ndarray(shape= a.shape, buffer=a)))
print(a)

# print(l.main(1, a))
print(l.main(2, a))
# print(l.main(3, a))

# l.main(2, np.ones[6]).get()
# print(t)
