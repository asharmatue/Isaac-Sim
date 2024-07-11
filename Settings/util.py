import time
import numpy as np

from Settings.settings import LOG_CONTEXT_SPACE, MEASURE_PERFORMANCE, DEBUG_LOG

def mod(a, n):
    res = a - np.floor(a/n) * n
    return res

def log(context:str, message:str, shifted:bool=False):
    spacing = ""
    if (len(context) < LOG_CONTEXT_SPACE):
        for _ in range(LOG_CONTEXT_SPACE - len(context)):
            spacing += " "
    if (shifted): 
        print(f"{spacing}{context} | {message}\n", end="")
    else: 
        print(f"{context}{spacing} | {message}\n", end="")

debug_contexts = set()

def add_debug_context(context:str):
    debug_contexts.add(context)

def debug_log(context:str, message:str, shifted:bool=False):
    if DEBUG_LOG or context in debug_contexts: 
        log(context, message, shifted)


total_performance = dict()
previous_time = time.time()

def performance_timestamp(label:str):
    if MEASURE_PERFORMANCE:
        global previous_time
        if label == "":
            previous_time = time.process_time()
            return
        
        time_difference = time.process_time() - previous_time
        if label in total_performance:
            total_performance[label] += time_difference
        else:
            total_performance[label] = time_difference

        log("performance_timestamp()", f"[{label}]")
        log("time", f"{round(time_difference, 4)}", True)
        log("total_time", f"{round(total_performance[label], 4)}", True)
        
        previous_time = time.process_time()