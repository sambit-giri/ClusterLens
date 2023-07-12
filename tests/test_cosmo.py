import numpy as np 
import ClusterLens

def test_InterfaceCCL():
	param = ClusterLens.par()
	cosmo = ClusterLens.InterfaceCCL(param)
	test1 = np.abs(cosmo.a_to_z(1)-0)<1e-3
	test2 = np.abs(cosmo.z_to_a(0)-1)<1e-3
	test3 = np.abs(cosmo.D(0)-1)<1e-3
	test4 = np.abs(cosmo.H(0)-100*param.cosmo.h)<1e-3
	test5 = np.abs(cosmo.z_to_cdist(10)-9600)<500
	assert test1 and test2 and test3 and test4 and test5 