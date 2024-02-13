import unittest
import numpy as np
from skypie.api import create_oracle, OracleType, PACKAGE_RESOURCES

class TestOracleAccuracy(unittest.TestCase):
    """
    This test compares the output of the ILP oracle with the SkyPIE oracle to verify the accuracy
    """

    def test_put_single_rep1(self):
        self.__test_put_single(1, "aws")

    def test_put_single_rep2(self):
        self.__test_put_single(2, "aws")

    def test_get_single_rep1(self):
        self.__test_get_single(1, "aws")
    
    def test_get_single_rep2(self):
        self.__test_get_single(2, "aws")

    def test_storage_single_rep1(self):
        self.__test_storage_single(1, "aws")

    def __test_put_single(self, replication_factor, region_selector):
        workload_dict = dict(
            #size=1, put={"aws-us-east-1":1}, get={"aws-us-east-1":0}, ingress={"aws-us-east-1":1}, egress={"aws-us-east-1":1}
            size=1, put={"aws-us-east-1":1}, get={"aws-us-east-1":0}
        )

        oracle_directory = PACKAGE_RESOURCES.get_default_oracle(min_replication_factor=replication_factor, region_selector=region_selector)

        cost_skypie, object_stores_skypie, app_assignments_skypie = self.__skypie_wrapper(workload_dict, oracle_directory, verbose=0)
        cost_ilp, object_stores_ilp, app_assignments_ilp = self.__ilp_wrapper(workload_dict, oracle_directory, verbose=0)

        print(f"SkyPIE: {cost_skypie}\n{object_stores_skypie}\n{app_assignments_skypie}")
        print(f"ILP: {cost_ilp}\n{object_stores_ilp}\n{app_assignments_ilp}")

        self.assertAlmostEquals(cost_skypie, cost_ilp)

    def __test_storage_single(self, replication_factor, region_selector):
        workload_dict = dict(
            #size=1, put={"aws-us-east-1":1}, get={"aws-us-east-1":0}, ingress={"aws-us-east-1":1}, egress={"aws-us-east-1":1}
            size=1, put={"aws-us-east-1":0}, get={"aws-us-east-1":0}
        )

        oracle_directory = PACKAGE_RESOURCES.get_default_oracle(min_replication_factor=replication_factor, region_selector=region_selector)

        cost_skypie, object_stores_skypie, app_assignments_skypie = self.__skypie_wrapper(workload_dict, oracle_directory, verbose=0)
        cost_ilp, object_stores_ilp, app_assignments_ilp = self.__ilp_wrapper(workload_dict, oracle_directory, verbose=0)

        print(f"SkyPIE: {cost_skypie}\n{object_stores_skypie}\n{app_assignments_skypie}")
        print(f"ILP: {cost_ilp}\n{object_stores_ilp}\n{app_assignments_ilp}")

        # Current storage cost of S3 Infrequent Access
        expected_cost = 0.0125 * replication_factor
        self.assertAlmostEquals(cost_skypie, expected_cost)
        self.assertAlmostEquals(cost_ilp, expected_cost)

    def __test_get_single(self, replication_factor, region_selector):
        workload_dict = dict(
            #size=1, put={"aws-us-east-1":1}, get={"aws-us-east-1":0}, ingress={"aws-us-east-1":1}, egress={"aws-us-east-1":1}
            size=1, put={"aws-us-east-1":0}, get={"aws-us-east-1":1}
        )

        oracle_directory = PACKAGE_RESOURCES.get_default_oracle(min_replication_factor=replication_factor, region_selector=region_selector)

        cost_skypie, object_stores_skypie, app_assignments_skypie = self.__skypie_wrapper(workload_dict, oracle_directory, verbose=0)
        cost_ilp, object_stores_ilp, app_assignments_ilp = self.__ilp_wrapper(workload_dict, oracle_directory, verbose=0)

        print(f"SkyPIE: {cost_skypie}\n{object_stores_skypie}\n{app_assignments_skypie}")
        print(f"ILP: {cost_ilp}\n{object_stores_ilp}\n{app_assignments_ilp}")

        self.assertAlmostEquals(cost_skypie, cost_ilp)

    @classmethod
    def __skypie_wrapper(cls, workload_dict, oracle_directory, verbose=0):
        ### Example to use SkyPIE
        # Find this example for the ILP baseline at examples/simple_ilp-baseline_example.py.

        # Create a SkyPIE oracle instance with the default arguments
        oracle = create_oracle(oracle_directory=oracle_directory, oracle_type=OracleType.SKYPIE, verbose=verbose)

        # Create a workload, specifying the workload features of applications accessing the object(s) per cloud region
        # This translates the workload features into a workload vector
        # and checks if the cloud region is supported by the loaded oracle.
        # See the DOC string for details!
        workload = oracle.create_workload_by_region_name(**workload_dict)
        # Alternatively, you can specify the workload features directly
        #workload = oracle.create_workload(size=1, put=[1,0,0], get=[0,1,0], ingress=[0,0,0], egress=[0,0,0])

        # Query the oracle for the optimal scheme
        decisions = oracle.query(w=workload, translateOptSchemes=True)

        cost, decision = decisions[0]
        # Object stores that store the object of the given workload, all of these need to receive updates of the object
        object_stores = decision.replication_scheme.object_stores
        # Assignment which cloud region sends get requests to which object store
        app_assignments = { a.app: a.object_store for a in decision.replication_scheme.app_assignments}

        #print(f"Optimal placement: {cost}\n{object_stores}\n{app_assignments}")
        return cost, object_stores, app_assignments

    @classmethod
    def __ilp_wrapper(cls, workload_dict, oracle_directory,*, verbose=0):
        ### Example to use ILP (SpanStore)
        # Find this example for the SkyPIE oracle at examples/simple_skypie_example.py.

        latency_slo = None
        latency_file_object_size = 41943040
        oracle_impl_args = {
            "latency_slo": latency_slo,
            "network_latency_file": PACKAGE_RESOURCES.network_latency_files[latency_file_object_size],
            # Set by default, but can be changed
            #"networkPriceFileName": PACKAGE_RESOURCES.networkPriceFileName, 
            #"storagePriceFileName": PACKAGE_RESOURCES.storagePriceFileName
        }

        # Create a ILP-based optimizer akin to the specified SkyPIE oracle
        oracle = create_oracle(oracle_directory=oracle_directory, oracle_type=OracleType.ILP, verbose=verbose, oracle_impl_args=oracle_impl_args)

        # Create a workload, specifying the workload features of applications accessing the object(s) per cloud region
        # This translates the workload features into a workload vector
        # and checks if the cloud region is supported by the loaded oracle.
        # See the DOC string for details!
        workload = oracle.create_workload_by_region_name(**workload_dict)
        # Alternatively, you can specify the workload features directly
        #workload = oracle.create_workload(size=1, put=[1,0,0], get=[0,1,0], ingress=[0,0,0], egress=[0,0,0])

        # Query the oracle for the optimal scheme
        decisions = oracle.query(w=workload, translateOptSchemes=True)

        cost, decision = decisions[0]
        # Object stores that store the object of the given workload, all of these need to receive updates of the object
        object_stores = decision.objectStores
        # Assignment which cloud region sends get requests to which object store
        app_assignments =  decision.assignments

        #print(f"Optimal placement: {cost}\n{object_stores}\n{app_assignments}")
        return cost, object_stores, app_assignments
    
if __name__ == '__main__':
    unittest.main()