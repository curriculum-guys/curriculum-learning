from curriculum_learning.logs import LogsHandler


class CurriculumLogs(LogsHandler):
    def __init__(self):
        super().__init__(interface="curriculum")
