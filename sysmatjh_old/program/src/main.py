import numpy as np
import matplotlib.pyplot as plt
from intersections import intersections

# Instantiate an instance of the library object
lib_hook = intersections()

# The grid runs from -n to n. Each gridline has a spacing of delta.
grid_size     = 6
delta         = 1.0
line_count    = 0
scan_start    = 89
scan_end      = 92   
scan_step     = 1.0

# Define the scanning angles

result = lib_hook.main(grid_size, delta, line_count, scan_start, scan_end, scan_step)
# print(np.shape(result))
#print(np.round(result[0], 5))
print(result[1])
# for i in range(len(result)):
#   print (result[i])
  # s = int(np.sqrt(len(result[i])))
  # m = np.reshape(result[i], (s,s))
  # m_rounded = np.round(m, 2)
  # print(np.matrix(m_rounded), "\n\n\n")





# def f(x, i, angle):
#     slope = np.tan(np.deg2rad(90) + np.deg2rad(angle))
#     delta_y = delta * i / np.cos(np.deg2rad(90) - np.deg2rad(angle))
#     return slope * x + delta_y

# zoom_out = 10
# # Plotting stuff
# for angle in range(int(scan_start), int(scan_start)+1):
#   plt.figure(int(angle))

#   plt.xticks(range(-grid_size, grid_size+1))
#   plt.xlim([-grid_size -zoom_out, grid_size +zoom_out])
#   plt.yticks(range(-grid_size, grid_size+1))
#   plt.ylim([-grid_size -zoom_out, grid_size +zoom_out])
#   plt.grid(True)

#   result = lib_hook.main(grid_size, delta, line_count, angle, scan_end, scan_step)
#   x_cords = result.data[0]
#   y_cords = result.data[1]

#   ii = list(range(-line_count, line_count+1))

#   for i in range(line_count * 2 + 1):
#     xs = range(-grid_size-2*zoom_out, grid_size+1+2*zoom_out)
#     ys = [ f(x, ii[i], angle) for x in xs ]
#     plt.plot(xs, ys, color='grey')
#     plt.scatter(x_cords[i], y_cords[i], zorder=5, s=4)

#     # coords = list(zip(x_cords[i], y_cords[i]))
#     # print(coords, '\n')
#   plt.show()
#   # plt.savefig('figs/fig' + str(angle) + '.png', bbox_inches='tight')
#   # plt.clf()


# ## Hadarmard ## Ill-posed / well-posed 
# ## Inverse problem / direct problem
# ## Filtered back-projection

