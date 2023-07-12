import numpy as np 
import ClusterLens

def test_InterfaceCCL():
	param = ClusterLens.par()
	bias  = ClusterLens.GalaxyBias(param)
	test1 = np.abs(bias.bias(1, kind='sqrt')-np.sqrt(2))<1e-3
	test2 = np.abs(bias.bias(2, kind='sqrt')-np.sqrt(3))<1e-3
	assert test1 and test2