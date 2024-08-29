import sys
sys.path.append("./")
import argparse
import logging
from typing import Dict, Any
import torch
import secrets
from src.base_config import BaseConfig
from src.net_utils import sample_hyperparameters
from src.config_utils import get_config_type
from copy import deepcopy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_config_with_sampled_params(base_config: Dict[str, Any], sampled_params: Dict[str, Any]) -> Dict[str, Any]:
    """Update the configuration with sampled parameters."""
    updated_config = deepcopy(base_config)
    method = updated_config.get('currentMethod')
    
    for param, value in sampled_params.items():
        keys = param.split('.')
        current = updated_config['UQMethods'][method]
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return updated_config

def run_model(config: Dict[str, Any], use_random_search: bool):
    method = config.get('currentMethod')
    if method is None:
        raise ValueError("The 'currentMethod' key is missing in the configuration file.")
    
    config_type = get_config_type(method)
    config_obj = config_type(config, use_random_search)
    config_obj.train_and_run_model()

def main(config_path: str, num_runs: int, use_random_search: bool):
    try:
        base_config = BaseConfig.load_config(config_path)
        
        method = base_config.get('currentMethod')
        if method is None:
            raise ValueError("The 'currentMethod' key is missing in the configuration file.")
        
        if use_random_search:
            search_config = base_config['UQMethods'][method].get('parameter_search')
            if not search_config:
                raise KeyError(f"Unable to perform random search: 'parameter_search' was not defined in the config file for method '{method}'!")

            for i in range(num_runs):
                logger.info(f"\nStarting run {i+1} of {num_runs}")
                logger.info("Performing random search for hyperparameters")

                updated_config = deepcopy(base_config)
                updated_config['random_seed'] = secrets.randbelow(int(1e16))  # Generate a new random seed for each run
                rgen = torch.Generator().manual_seed(updated_config['random_seed'])
                
                sampled_params = sample_hyperparameters(search_config, rgen)
                updated_config = update_config_with_sampled_params(updated_config, sampled_params)
                try:
                    run_model(updated_config, use_random_search)
                except Exception as e:
                    logger.error(f"An error occurred in run {i+1}: {str(e)}")
                    continue
        else:
            logger.info("Using default configuration")

            run_model(base_config, use_random_search)
        
        logger.info("All runs completed")
    except Exception as e:
        logger.error(f"An error occurred in main function: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using either the passed configuration directly, or draw specified parameters randomly according to the `parameter_search` section")

    parser.add_argument("config_path", help="The path to the JSON config file.")
    group = parser.add_argument_group(title="random search", description="In order to enable using random search, set the flag and specify the number of iterations.")
    group.add_argument("-i", "--search_iterations", required=False, default=1, type=int, help="Specify the number of subsequent runs with randomly drawn parameters")
    group.add_argument("-s", "--use_random_search", required=False, default=False, action="store_true", help="Enable random search")

    args = parser.parse_args()

    if not args.use_random_search and args.search_iterations > 1:
        logger.error(f"Search iterations were specified ({args.search_iterations}), but the '--use_random_search | -s' flag was not set. Did you mean to use random search?")
        sys.exit(1)

    main(args.config_path, args.search_iterations, args.use_random_search)
