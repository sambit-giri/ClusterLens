import numpy as np 
import ClusterLens

def test_age_estimator():
	param = ClusterLens.param()
	t0 = ClusterLens.age_estimator(param, 0)
	assert np.abs(t0-13.74)<0.01