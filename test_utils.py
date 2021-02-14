from datetime import datetime

now = datetime.now()

class Day:
    def __getitem__(self, item):
        return datetime(2000, 1, item)

day = Day()
