import matplotlib.pyplot as plt

polygon = [(0, 0), (1, 1), (1, 0), (0, 1), (2, 0.5)]
partition = [[(1, 0.25), (2, 0.5), (1, 0.75)], [(0.5, 0.5), (0, 0), (0.8, 0.2), (1, 1)], [(0.8, 0.8), (0, 1), (0.5, 0.5)], [(0.8, 0.2), (1, 0), (1, 1)]]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot original polygon
polygon.append(polygon[0])  # Close the polygon
polygonX, polygonY = zip(*[(p[0], p[1]) for p in polygon])

ax[0].plot(polygonX, polygonY, 'b-')
ax[0].fill(polygonX, polygonY, 'lightblue', alpha=0.5)
ax[0].set_title('Original Polygon')
ax[0].set_aspect('equal')

# Plot convex partition
for poly in partition:
	poly.append(poly[0])  # Close the polygon
	polyX, polyY = zip(*[(p[0], p[1]) for p in poly])
	ax[1].plot(polyX, polyY)
	ax[1].fill(polyX, polyY, alpha=0.5)

ax[1].set_title('Convex Partition')
ax[1].set_aspect('equal')

plt.show()

