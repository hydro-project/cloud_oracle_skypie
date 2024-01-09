import numpy as np
import json
from typing import List

from skypie.oracle import Oracle
from skypie.util.my_dataclasses import Workload

def load_workloads(oracle: Oracle, *, input: str) -> List[Workload]:
    """
    Load workloads from a file and create a list of Workload objects.

    Args:
        oracle (Oracle): The Oracle object used to create the Workload objects.
        input (str): The path to the input file containing the workloads.

    Returns:
        List[Workload]: A list of Workload objects created from the input file.

    Raises:
        ValueError: If a region specified in the workload is not found in the application regions.
    """
    workloads = []

    # Load a list of workloads from a file
    with open(input, "r") as f:
        skipLines = False
        for i, line in enumerate(f.readlines()):
            if line.startswith("//") or len(line) == 0:
                continue
            
            workload = json.loads(line)

            parsedWorkload = dict(
                put = np.zeros(oracle.no_apps),
                get = np.zeros(oracle.no_apps),
                ingress = None,
                egress = None,
                size=0
            )

            # Parse accesses of application regions
            for accessCategory in ["put", "get"]:
                if accessCategory in workload:
                    for region, count in workload[accessCategory].items():
                        if region in oracle.__applicationRegions:
                            parsedWorkload[accessCategory][oracle.__applicationRegions[region]] = count
                        else:
                            raise ValueError("Region " + region + f" in workload {accessCategory} spec. not found in application regions!\n" + "Application regions: " + str(oracle.__applicationRegions) + "\n" + Oracle.get_workload_file_format())
                else:
                    print(f"WARNING: No {accessCategory} workload specified for workload {i}, assuming 0 for all regions!")
                
            # Parse network volume of application regions
            for networkCategory in ["ingress", "egress"]:
                if networkCategory in workload:
                    # Allocate list
                    parsedWorkload[networkCategory] = np.zeros((oracle.no_apps))
                    # Fill list with workload of application region at its index
                    for region, count in workload[networkCategory].items():
                        if region in oracle.__applicationRegions:
                            parsedWorkload[networkCategory][oracle.__applicationRegions[region]] = count
                        else:
                            raise ValueError("Region " + region + f" in workload {networkCategory} spec. not found in application regions!\n" + "Application regions: " + str(oracle.__applicationRegions) + "\n" + Oracle.get_workload_file_format())
                else:
                    print(f"WARNING: {networkCategory} not explicitly specified in workload. Computing it based on accesses and object size!")
                

            if "size" in workload:
                parsedWorkload["size"] = workload["size"]
            else:
                print(f"WARNING: No size specified for workload {i}, assuming 0!")

            workloads.append(oracle.create_workload(**parsedWorkload))

    if len(workloads) < 1:
        print("Input file does not contain any workloads!")

    return workloads
