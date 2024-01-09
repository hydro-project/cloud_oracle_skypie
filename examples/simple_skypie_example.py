#import skypie.api as skypie
from skypie.api import create_oracle, OracleType, PACKAGE_RESOURCES

### Example to use SkyPIE
# Find this example for the ILP baseline at examples/simple_ilp-baseline_example.py.

# Path to the oracle files.
# This oracle was precomputed for placement within all of AWS
# under replication factor min=2 to max=2 and no latency SLO.
oracle_directory = PACKAGE_RESOURCES.get_default_oracle(min_replication_factor=2, region_selector="aws")

# Create a SkyPIE oracle instance with the default arguments
oracle = create_oracle(oracle_directory=oracle_directory, oracle_type=OracleType.SKYPIE, verbose=1)

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
object_stores = decision.replication_scheme.object_stores
# Assignment which cloud region sends get requests to which object store
app_assignments = { a.app: a.object_store for a in decision.replication_scheme.app_assignments}

print(f"Optimal placement: {cost}\n{object_stores}\n{app_assignments}")