Folder：
          data: all the data sets used in the experiment，including 12 txt files

File：
          Algorithms.py: compared algorithms, including：
		-kmeans():   K-means
		-agglomerativeClustering:   HAC
		-dbscan():    DBSCA
		-OPTICS():   OPTICS
		-SCA_clustering():   SCA

          SCA2.py: the proposed SCA2，including the following function：
		-calculate_neighborhood(): acquire the neighbors for each data point based on kNN or epison-radius
		-labeling(): label the data point
		-PSOClusteringAlgorithm(): the main framework of SCA2

          PublicFunctions.py: public operation，including：
		-select_real_world_datasets():   select a real world data set
		-select_synthetic_datasets():   select a synthetic data set
		-select_file():   select the type of data set
		-readRawDataFromFile():   read the raw data from the file
		-getDistance():   calculate the Euclidean distance between two points
		-getAverageDistance():   calculate the average distance between points
		-calcFMeasure():   calculate F-Measure
		-calNMI():   calculate NMI
		-calARI():   calculate ARI
		-calValidator():   calculate F-measure、NMI and ARI
		-drawClusteringResultGraph():    draw the clustering figure

          SCA.py:  the source code of SCA

          newrb.py: implement the newrb() function in Matlab using Python. We only implement the major function designrb() in newrb().
