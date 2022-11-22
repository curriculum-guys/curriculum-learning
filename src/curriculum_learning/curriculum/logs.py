from datetime import datetime
import logging
from data_interfaces.utils import create_dirs, get_root_dir

class CurriculumLogs:
    def create(self, name):
        self.name = name
        self.__logger = logging.getLogger(f'{name}_curriculum')
        self.__logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.__filename)
        fh.setLevel(logging.DEBUG)
        self.__logger.addHandler(fh)

    @property
    def __logs_dir(self):
        root_dir = get_root_dir()
        log_dir = f'logs/{self.name}/curriculum'
        create_dirs(root_dir, log_dir)
        return log_dir

    @property
    def __filename(self): 
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        return f'{self.__logs_dir}/{date_time}.log'

    def update(self, gen, config):
        for key, value in config.items():
            self.__logger.debug(f'[{gen}] {key.upper()} - {value}')
        self.__logger.debug('-' * 10)
