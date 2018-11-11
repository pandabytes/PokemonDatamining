def elapsedTime(func):
    def wrapper(*args, **kwargs):
        timeStart = time.time()
        result = func(*args, **kwargs)
        elapsedTime = time.time() - timeStart
        print("Elapsed time:", elapsedTime, "seconds")
        return result
    return wrapper