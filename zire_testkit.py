from argparse import ArgumentParser
import yaml
from modules import binary_reader, fitter, plotter, build_json, csv_reader
from modules.ZLogger import get_logger
from modules import config_loader
from modules.build_channel_map import build_channel_map

log = get_logger()
default_yaml = "testkit_config.yaml"

TASK_MODULES = {
    "build_json": build_json,
    "csv_reader": csv_reader,
    "binary_reader": binary_reader,
    "fitter": fitter,
    "plotter": plotter
}

def main():
    """Parses command-line arguments to get the YAML input file name."""
    parser = ArgumentParser(description="Process a YAML configuration file for Zire test kit data processing.")
    parser.add_argument("input_file", type=str, 
                        nargs="?",                  # if no argument is provided, it will use the default
                        default=default_yaml,       # default value
                        help="Path to the input YAML file.")
    args = parser.parse_args()

    config = read_yaml(args)
    config_loader.set_config(config)

    for task in config['tasks']:
        
        if task['enabled'] and task["name"] != "channel_mapping":
            module = TASK_MODULES.get(task['name'])
            if module is None:
                log.error(f"Task '{task['name']}' is not recognized or not implemented.")
                continue
            
            module.perform_task(task)


def read_yaml(args):
    # filename from the YAML file
    filename = args.input_file

    # check if the file exists and is a valid YAML file
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        log.error(f"Error: The file '{filename}' was not found.")
        exit(1)
    except yaml.YAMLError as e:
        log.error(f"Error: Failed to parse YAML file '{filename}'.\n{e}")
        exit(1)

# The main function is called when the script is executed directly
if __name__ == "__main__":
    main()