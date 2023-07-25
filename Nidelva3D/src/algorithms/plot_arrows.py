import matplotlib.pyplot as plt

# Create a figure and axes
fig, ax = plt.subplots()

# Set the limits of the plot
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

# Plot an arrow
arrow = plt.Arrow(2, 2, 1, 1, width=0.5, color='r')
ax.add_patch(arrow)

# Set the aspect ratio
ax.set_aspect('equal')

# Show the plot
plt.show()

