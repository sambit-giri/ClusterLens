import numpy as np 
import ClusterLens

def test_InterfaceCCL():
	param = ClusterLens.param()
	cosmo = ClusterLens.InterfaceCCL(param)
	test1 = np.abs(cosmo.a_to_z(1)-0)<1e-3
	test2 = np.abs(cosmo.z_to_a(0)-1)<1e-3
	assert test1