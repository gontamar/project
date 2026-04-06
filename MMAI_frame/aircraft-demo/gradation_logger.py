import time
import psutil
import logging
import os
from datetime import datetime
 
 
class GradationLogger:

    def __init__(self, logfile=None):
 
        self.stages = {}
        self.process = psutil.Process(os.getpid())
 
        # -----------------------------
        # CREATE UNIQUE LOG FILE
        # -----------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
 
        if logfile is None:
            logfile = f"performance_logs_{timestamp}.txt"
 
        # ensure logs folder (optional but clean)
        os.makedirs("logs", exist_ok=True)
        logfile_path = os.path.join("logs", logfile)
 
        # create new file
        with open(logfile_path, "w") as f:
            f.write("===== PERFORMANCE LOGS =====\n\n")
 
        # -----------------------------
        # LOGGER SETUP
        # -----------------------------
        self.logger = logging.getLogger(f"gradation_logger_{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
 
        handler = logging.FileHandler(logfile_path)
 
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%H:%M:%S"
        )
 
        handler.setFormatter(formatter)
 
        self.logger.handlers = []
        self.logger.addHandler(handler)
 
        # store path (optional for debugging)
        self.logfile_path = logfile_path
 
        print(f"[Logger Initialized] Writing to: {logfile_path}")
 
    # def __init__(self, logfile="performance_logs.txt"):
 
    #     self.stages = {}
    #     self.process = psutil.Process(os.getpid())
 
    #     # fresh file every run
    #     with open(logfile, "w") as f:
    #         f.write("===== PERFORMANCE LOGS =====\n\n")
 
    #     self.logger = logging.getLogger("gradation_logger")
    #     self.logger.setLevel(logging.INFO)
    #     self.logger.propagate = False
 
    #     handler = logging.FileHandler(logfile)
 
    #     formatter = logging.Formatter(
    #         "%(asctime)s | %(message)s",
    #         datefmt="%H:%M:%S"
    #     )
 
    #     handler.setFormatter(formatter)
 
    #     self.logger.handlers = []
    #     self.logger.addHandler(handler)
 
 
    def log(self, message):
        self.logger.info(message)
 
 
    def start_stage(self, name):
 
        self.stages[name] = {
            "start": time.perf_counter(),
            "mem_start": self.process.memory_info().rss
        }
 
        self.log(f"[START] {name}")
 
 
    def end_stage(self, name):
 
        if name not in self.stages:
            self.log(f"[WARNING] end_stage called without start_stage: {name}")
            return
 
        end = time.perf_counter()
        mem = self.process.memory_info().rss
 
        stage = self.stages[name]
 
        duration = end - stage["start"]
        mem_used = mem - stage["mem_start"]
 
        self.log(f"Duration: {duration:.3f} sec")
        self.log(f"Memory Change: {mem_used/1024/1024:.2f} MB")
        self.log(f"[END] {name}\n")
 
 
    def start_pipeline(self):
 
        self.pipeline_time = time.perf_counter()
 
        self.log("========== PIPELINE START ==========")
 
 
    def end_pipeline(self):
 
        total = time.perf_counter() - self.pipeline_time
 
        self.log(f"TOTAL RESPONSE TIME: {total:.3f} sec")
        self.log("========== PIPELINE END ==========\n")
 
 
    def log_chat_memory(self, session_memory):
 
        messages = session_memory.get_messages()
    
        count = len(messages)
        mem = sum(len(str(m)) for m in messages)
    
        self.log("CHAT MEMORY")
        self.log(f"Messages Stored: {count}")
        self.log(f"Approx Memory: {mem/1024:.2f} KB\n")
 
            
 
# global instance
logger = GradationLogger()
 