from typing import Dict, Any
import os
from skypie.oracle import Oracle
from skypie.util.my_dataclasses import OracleType
from skypie.util.util import PACKAGE_RESOURCES, Timer

__all__ = ["create_oracle", "OracleType", "get_AVAILABLE_ORACLE_IMPLEMENTATIONS", "get_default_oracle_impl_args", "PACKAGE_RESOURCES", "Timer"]

def create_oracle(*, oracle_directory:str, oracle_type: OracleType, scenario_name:str = "default", verbose: int = 0, oracle_impl_args: Dict[str, Any] = dict(), verify_experiment_args: bool = True) -> Oracle:
    """
    Creates an instance of the Oracle, which is used to compute object placement decisions for a given input file and scenario.

    # Arguments
    - oracle_file (str): The path to the input file containing the precomputed oracle data.
    - oracle_type (OracleType): The type of oracle to be used for prediction. Must be one of the values defined in the OracleType enum.
    - scenario_name (str, optional): The name of the scenario to be used when several oracles have been precomputed in the same input file. Defaults to "default".
    - verbose (int, optional): The level of verbosity for logging. 0 means no logging, 1 means basic logging, and 2 means detailed logging. Defaults to 0.
    - oracle_impl_args (Dict[str, Any], optional): A dictionary of implementation-specific arguments to be passed to the oracle implementation. Defaults to an empty dictionary. An empty dictionary means that the default values of the implementation will be used. The default arguments can be retrieved using the get_default_oracle_impl_args function, then be modified and passed to this function.
    - verify_experiment_args (bool, optional): A flag to verify the experiment arguments against the oracle implementation arguments. Verification fails if experiment arguments are overwritten and do not match. Defaults to False.

    # Returns
        Oracle: An instance of the Oracle, which is used to compute object placement decisions for a given input file and scenario.

    # Note:
     -  The oracle type "OracleType.ILP" requires input files for storage prices, network prices and optionally for network latency.
        There are default files for these in the package, which can be retrieved using the PACKAGE_RESOURCES object.
        Also, for "OracleType.ILP" the latency SLO can be set using the oracle_impl_args dictionary.
        
     -  The oracle type "OracleType.SKYPIE" assumes a SLO during its offline precomputation!
        You have to choose the SLO during precomputation and use the according oracle file.
    """

    assert oracle_type in OracleType, f"Unknown oracle type {oracle_type}"
    oracle_type_value = oracle_type.value

    input_file_name = os.path.join(oracle_directory, "stats.proto.bin")
    # Verify that the directory exists
    assert os.path.isdir(oracle_directory), f"Directory {oracle_directory} does not exist"
    # Verify that the file exists
    assert os.path.isfile(input_file_name), f"File {input_file_name} does not exist"

    implArgs = oracle_impl_args

    # Read the experiment arguments to check for contradictions with the oracle implementation args
    experiment_args_file_name = os.path.join(oracle_directory, "experiment.json")
    if os.path.isfile(experiment_args_file_name) and verify_experiment_args:
        import json
        contradictions = dict()
        experiment_args = json.load(open(experiment_args_file_name, "r"))
        for key, value in oracle_impl_args.items():
            if key in experiment_args:
                if experiment_args[key] != value:
                    contradictions[key] = (experiment_args[key], value)

        if len(contradictions) != 0:
            print(f"Contradictions between oracle implementation args and experiment args (key: oracle arg, provided arg): {contradictions}")

            # Raise an error if there are critical contradictions
            critical_contradictions = {key:value for key, value in contradictions.items() if value[1] is None and value[2] is not None or value[2] is None and value[1] is not None}
            assert len(critical_contradictions) == 0, f"Critical contradictions between oracle implementation args and experiment args (key: oracle arg, provided arg): {critical_contradictions}"
    
    oracle, optimizer = Oracle.setup_oracle(inputFileName=input_file_name, scenarioName=scenario_name, optimizer=[oracle_type_value], implArgs=implArgs, verbose=verbose)

    assert len(optimizer) == 1, f"Expected exactly one optimizer, but got {optimizer}"

    oracle.prepare_scenario(scenario_name=scenario_name, optimizer_type=optimizer[0], compact=False)

    return oracle

def get_AVAILABLE_ORACLE_IMPLEMENTATIONS():
    """
    Returns a list of all available oracle implementations.
    """
    return Oracle.AVAILABLE_ORACLE_IMPLEMENTATIONS()

def get_default_oracle_impl_args(oracle_type: OracleType) -> Dict[str,Any]:
    """
    Returns the default arguments for a given oracle implementation for customizing.
    """
    assert oracle_type in Oracle.AVAILABLE_ORACLE_IMPLEMENTATIONS, f"Unknown oracle type {oracle_type}"
    Oracle_Impl = Oracle.AVAILABLE_ORACLE_IMPLEMENTATIONS[oracle_type]

    return Oracle_Impl.get_default_args()