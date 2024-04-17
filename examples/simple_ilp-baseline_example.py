#import skypie.api as skypie
from skypie.api import create_oracle, OracleType, PACKAGE_RESOURCES

### Example to use ILP (SpanStore)
# Find this example for the SkyPIE oracle at examples/simple_skypie_example.py.

# Path to the oracle files.
# This oracle was precomputed for placement within all of AWS
# under replication factor min=2 to max=2 and no latency SLO.
oracle_directory = PACKAGE_RESOURCES.get_default_oracle(min_replication_factor=2, region_selector="aws")

latency_slo = None
latency_file_object_size = 41943040
oracle_impl_args = {
    "latency_slo": latency_slo,
    "network_latency_file": PACKAGE_RESOURCES.network_latency_files[latency_file_object_size],
    #"ignore_considered_scenario": True,
    #"application_regions": ['gcp-europe-west1-b', 'azure-eastus', 'azure-westus', 'gcp-us-east1-b', 'gcp-us-west1-a', 'azure-westeurope', 'aws-us-east-1', 'aws-eu-west-1', 'azure-westus']
    "region_selector": "europe|us",
    "object_store_selector": "General Purpose|Standard Storage|Hot",
    # Set by default, but can be changed
    #"networkPriceFileName": PACKAGE_RESOURCES.networkPriceFileName, 
    #"storagePriceFileName": PACKAGE_RESOURCES.storagePriceFileName
}

# Create a ILP-based optimizer akin to the specified SkyPIE oracle
oracle = create_oracle(oracle_directory=oracle_directory, oracle_type=OracleType.ILP, verbose=1, oracle_impl_args=oracle_impl_args, verify_experiment_args=False)

# Create a workload, specifying the workload features of applications accessing the object(s) per cloud region
# This translates the workload features into a workload vector
# and checks if the cloud region is supported by the loaded oracle.
# See the DOC string for details!
workload = oracle.create_workload_by_region_name(size=1, put={"aws-us-east-1":1}, get={"aws-us-east-1":1}, ingress={"aws-us-east-1":1}, egress={"aws-us-east-1":1})
# Alternatively, you can specify the workload features directly
#workload = oracle.create_workload(size=1, put=[1,0,0], get=[0,1,0], ingress=[0,0,0], egress=[0,0,0])

# Query the oracle for the optimal scheme
decisions = oracle.query(w=workload, translateOptSchemes=True)

cost, decision = decisions[0]
# Object stores that store the object of the given workload, all of these need to receive updates of the object
object_stores = decision.objectStores
# Assignment which cloud region sends get requests to which object store
app_assignments =  decision.assignments

print(f"Optimal placement: {cost}\n{object_stores}\n{app_assignments}")
print("DONE")