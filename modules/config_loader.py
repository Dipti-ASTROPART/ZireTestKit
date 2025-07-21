# modules/config_loader.py
_config = {}

def set_config(config):
    global _config
    _config = config

def get_channel_map_paths():
    try:
        channel_map_config = next(
            task for task in _config['tasks']
            if task.get("name") == "channel_mapping"
        )
        return {
            "pst_map_file": channel_map_config["pst_map_file"],
            "calog_map_file": channel_map_config["calog_map_file"],
            "acs_map_file": channel_map_config["acs_map_file"]
        }
    except StopIteration:
        raise ValueError("Channel mapping configuration not found in YAML.")