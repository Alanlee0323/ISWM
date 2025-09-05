import functools
import traceback

def safe_process(error_value=None):
    """安全處理裝飾器，用於錯誤處理"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                return error_value
        return wrapper
    return decorator

def timing_debug(func):
    """計時裝飾器，用於性能分析"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper