class BaseException(Exception):
    pass

class StorePathFileExist(BaseException):
    def __init__(self, fp):
        self.msg = "%s already exists!"