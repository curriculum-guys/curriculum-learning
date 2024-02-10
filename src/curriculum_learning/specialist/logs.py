from curriculum_learning.logs import LogsHandler


class SpecialistLogs(LogsHandler):
    def __init__(self):
        super().__init__(interface="specialist")