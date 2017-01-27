import shapely
import networkx as nx
import json
import community
#import geojson
import pdb
import numpy as np
from scipy.spatial import distance
from community import best_partition

def calculate_degree(graph):
	print ("Calculating degree...")
	g = graph
	deg = nx.degree(g)
	nx.set_node_attributes(g,'degree',deg)
	return g, deg

# load data for 47 buildings
with open('../data/geo_eg.json') as f:
    data = json.load(f)
total = []
# compute centroid of each buildings
for i,feature in enumerate(data['features']):
	if (i > 0):
		print("Start loading building information")
		#for each cube, find the collections of all points
		convexs = set()
		centroid = []

		for surfaces in feature['geometry']['coordinates']:
			surface = surfaces[0][0:4]
			for p in surface:
				convexs.add(tuple(p))
		convexs = list(convexs)
		centroid = [sum([x[0] for x in convexs])/8.0,sum([x[1] for x in convexs])/8.0,sum([x[2] for x in convexs])/8.0]
		total.append(centroid)

X = np.matrix(total)

# compute pairwise distance(d2/angle) of 47 * 47 
Y = distance.pdist(X,'euclidean')
Z = distance.cdist(X,X,'euclidean')
np.histogram(Y)

# select threshold of distance 
H = np.zeros(Z.shape)
d = np.percentile(Y,10)
# construct adjancy 
for x in range(len(Z)):
	for y in range(len(Z)):
		if ((x!=y) and (Z[x][y] < d)):
			H[x][y] = Z[x][y]

# construct grpahs
g =nx.from_numpy_matrix(Z)
num_nodes = nx.number_of_nodes(g)
g, deg = calculate_degree(g)

#detect numbers of communities!
part = community.best_partition(g)

pdb.set_trace()
print("break point")
