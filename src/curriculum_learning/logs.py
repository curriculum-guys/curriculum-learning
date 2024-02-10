from datetime import datetime
import logging
from data_interfaces.utils import create_dirs, get_root_dir

class LogsHandler:
    def __init__(self, interface):
        self.interface = interface

    def create(self, name):
        self.name = name

        self.root_dir = get_root_dir()
        logs_dir = f"logs/{self.name}/{self.interface}"
        self.logs_dir = f"{self.root_dir}/{logs_dir}"
        create_dirs(self.root_dir, logs_dir)

        self.__logger = logging.getLogger(f'{name}_{self.interface}')
        self.__logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.__filename)
        fh.setLevel(logging.DEBUG)
        self.__logger.addHandler(fh)

    @property
    def __filename(self): 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        return f'{self.logs_dir}/{date_time}.log'

    def update(self, gen, config):
        for key, value in config.items():
            self.__logger.debug(f'[{gen}] {key.upper()} - {value}')
        self.__logger.debug('-' * 10)
