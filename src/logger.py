import logging
import sys
import json
import time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": time.time(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
