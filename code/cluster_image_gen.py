# Basic Stuff
import numpy as np
import json
from prettytable import PrettyTable
import itertools
import pandas as pd
import operator
# Geo Stuff
import shapely
from shapely.geometry import Polygon

# Plot Stuff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline 

# Machine Learning Stuff
from sklearn import cluster,metrics
from scipy.spatial import distance
# from sklearn.model_selection import GridSearchCV

# turn warnings off
import warnings
warnings.filterwarnings('ignore')

#debug stuff
import pdb
from IPython.core.debugger import Tracer

plot_ns = range(1,515)
for plot_n in plot_ns:
	#load data
	with open('../data/geometry/' +str(plot_n)+'.json') as f:
		data = json.load(f)

##
	centroid_alles = []
	polygon_alles = []
	points_alles = []
	for i,feature in enumerate(data['features']):
	    if (i > 0):
	#for each cube, find the collections of all 8 points
	        convexs = set()
	        centroid = []

	        for surfaces in feature['geometry']['coordinates']:
	            
	            surface = surfaces[0]
	            surface_z = set(x[2] for x in surface)
	            if len(surface_z) == 1:
	                surface_2d = list((x[0],x[1]) for x in surface)
	            for p in surface:
	                convexs.add(tuple(p))
	        convexs = list(convexs)
	        centroid = [sum([x[0] for x in convexs])/8.0,sum([x[1] for x in convexs])/8.0,sum([x[2] for x in convexs])/8.0]
	        centroid_alles.append(centroid)
	        points_alles.append(list(itertools.chain.from_iterable(surface_2d[0:4])))
	        polygon = Polygon(surface_2d)
	        polygon = shapely.geometry.polygon.orient(polygon)
	        polygon_alles.append(polygon)
	X = np.matrix(centroid_alles)
	X_points = np.matrix(points_alles)
## compute pairwise distance 
	adj_poly = np.zeros((47,47))
	for c,c_poly in enumerate(polygon_alles):
	    for a,a_poly in enumerate(polygon_alles):
	        adj_poly[c,a] = c_poly.distance(a_poly)
	#print(adj_poly)	
# compute pairwise distance between centroids
	Y = distance.pdist(X,'euclidean')# pairwise distance 平直的
	Z = distance.cdist(X,X,'euclidean')# 47 by 47 

	Y_points = distance.pdist(X_points,'euclidean')
	Z_points = distance.cdist(X_points,X_points,'euclidean')
	#2D matrix
	X_2D = X[:,0:2]
	Y_2D = distance.pdist(X_2D,'euclidean')# pairwise distance 平直的
	Z_2D = distance.cdist(X_2D,X_2D,'euclidean')# 47 by 47  
	
	pw_dis = list(adj_poly[np.triu_indices(47)])
	pw = [y for y in pw_dis if y != 0.0]
##
	min_per = 2
	max_per = 5
	percentile = list(np.arange(min_per,max_per,0.1))
	min_samples_ = [2,3,4]
	eps_ = []
	para_dist = []

	t = PrettyTable(['percentile','eps','min_sample','silhouette'])

	# for val in percentile:
	#     eps_.append(np.percentile(pw,val))
	si_max = -1
	for min_samples in min_samples_:
	    for val in percentile:
	        eps = np.percentile(pw,val)
	        dbscan = cluster.DBSCAN(
	            eps=eps,
	            min_samples=min_samples,
	            metric='precomputed')
	        labels = dbscan.fit_predict(adj_poly)
	        if(len(set(labels)) == 1):
	            SI_score = 'NA'
	        else:
	            SI_score = metrics.silhouette_score(adj_poly,labels,metric="precomputed")
	            t.add_row([val,eps,min_samples,SI_score])
	            para_dist.append({'eps':eps,'min_sample':min_samples,'SI':SI_score})
	            if (SI_score > si_max):
	                si_max = SI_score
	                para_set_max = [val,eps,min_samples]
	        

	#print("Best parameter",para_set_max,"with Silhouette Index of",si_max)
	#print (t.get_string(sortby="silhouette",reversesort=True))
	newlist = sorted(para_dist, key=lambda k: k['SI'],reverse=True) 
	good_para = newlist[0:9]
	#print (t.get_string(sortby="silhouette",reversesort=True))
	fig = plt.figure()
	print(good_para)
	ind_plot = [331,332,333,334,335,336,337,338,339]
##
	for ind,val in enumerate(good_para):
	   
	    ax = fig.add_subplot(ind_plot[ind])
	    eps = val['eps']
	    min_samples = val['min_sample']
	    SI_score = val['SI']
	    dbscan = cluster.DBSCAN(eps=eps,min_samples=min_samples,metric='precomputed')
	    labels = dbscan.fit_predict(adj_poly)
	    
	    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
	    core_samples_mask[dbscan.core_sample_indices_] = True
	    
	    n_clusters_ = len(set(labels)) #exclusively consider the noisy samples as a cluster
	    
	##
	    unique_labels = set(labels)
	    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

	    for k, col in zip(unique_labels, colors):
	        if k == -1:
	            # Black used for noise.
	            col = 'k'

	        class_member_mask = (labels == k)

	        xy = X_2D[class_member_mask & core_samples_mask]
	        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	                 markeredgecolor='k', markersize=4.5)

	        xy = X_2D[class_member_mask & ~core_samples_mask]
	        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	                 markeredgecolor='k', markersize=3)
	##
	    for i,poly in enumerate(polygon_alles):
	        x,y = poly.exterior.xy
	        ax.plot(x, y, color='#6699cc', alpha=0.7,
	        linewidth=1, solid_capstyle='round', zorder=2)
	        
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	    ax.set_frame_on(False)
	    #ax.set_title(" Title", size=8)
	    ax.set_title('SI score: %s' % (SI_score),size=8)
	    
	fig.savefig('../data/clustering/'+ str(plot_n) +'.png')  







