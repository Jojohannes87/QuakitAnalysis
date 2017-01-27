# QuakitAnalysis

###Introduction
We have collected 500+ design proposals from qua-kit tool. Each design is an urban layout and it is represented in geometry format (in geojson). Within each design, 47 cubes(builings) are placed according to designers' intention. This project implements spatial data mining and attempts to detect patterns of their geometries and translates into urban design languages. As geometry need to be mapped to urban design languages esp. urban forms(open spaces, road, entrance/exit etc.), shapes need to be labelled with human input to train machine learning algorithm.
It could be considered as the evaluation part of citizen design science. 

###Goal
* Nearest neighourhood detection
* Feature extraction
* Image segmentation (potential)
* clustering of designs


###Installation
* /data contains an example of an geometry example
* /code network.ipynb implements spatial data clustering using DBSCAN 

###Tools&Library
* python: sklearn, shapely
* HTML/javascript:
* Crowdflowers/Amazon Mechnical Turk


###Todo
* Nearest Neighbourhood Detection
* Crowdsourcing 