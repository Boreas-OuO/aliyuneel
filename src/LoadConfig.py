import json
from src import LogConfig
class LoadConfig():
    def __init__(self,
                 prefix_path,
                 configs_path) -> None:
        self.prefix = json.load(open(prefix_path, 'r'))
        self.configs = json.load(open(configs_path, 'r'))
        self.row = list(self.prefix.keys())

    

        
    
        