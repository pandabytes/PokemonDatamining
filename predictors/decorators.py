import time

def elapsedTime(func):
    ''' '''
    def wrapper(*args, **kwargs):
        timeStart = time.time()
        result = func(*args, **kwargs)
        elapsedTime = time.time() - timeStart
        print("Function \"{0}\" took {1:.2f} seconds to complete".format(func.__name__, elapsedTime))
        return result
    return wrapper