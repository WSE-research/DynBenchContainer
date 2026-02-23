from time import time, sleep
from collections import defaultdict as ddict
from typing import Hashable


_last_time = ddict(float)


def wait_time(wait: float, timer_ID: Hashable=None, asynchronous: bool=True) -> bool:
    """
    Measure a minimum time interval between successive calls with the same timer_ID.
    Useful for rate limiting and throttling operations (e.g., API calls or resource-intensive tasks).
    
    This function is NOT thread-safe and should not be used in multi-threaded environments.
    
    Args:
        wait: Minimum time interval (in seconds) to wait.
        timer_ID: Optional identifier for this timer. If None, uses a default timer.
        asynchronous: If True, returns immediately with False if insufficient time has passed.
            If False, blocks execution until the wait period is satisfied.

    Returns:
        bool: True if enough time has passed since the last call (or on first use),
            False if the wait period hasn't elapsed yet (only when asynchronous=True).

    Raises:
        TypeError: If wait is not a number or timer_ID is not hashable.
        ValueError: If wait is negative.
    """
    global _last_time
    
    if not isinstance(wait, (int, float)):
        raise TypeError('wait parameter must be a number')
    if wait < 0:
        raise ValueError('wait parameter cannot be negative')
    if not isinstance(timer_ID, Hashable):
        raise TypeError('timer_ID must be a hashable type')
    
    now = time()

    # Check if this timer has been used before
    if timer_ID in _last_time:
        # If insufficient time has passed
        if now - _last_time[timer_ID] < wait:
            if asynchronous:
                return False
            else:
                # Block until enough time has passed
                sleep(max(0, wait - (now - _last_time[timer_ID]))) 
        # Record the current time as the new reference point
        _last_time[timer_ID] = time()
        return True
    else:
        # First time this timer is used, record the current time
        _last_time[timer_ID] = now
        return True


def timer_reset(timer_ID: Hashable=None) -> bool:
    """
    Reset a specific timer by clearing its recorded timestamp.
    Subsequent calls with this timer_ID will behave as if it's being used for the first time.
    
    This function is NOT thread-safe and should not be used in multi-threaded environments.
    
    Args:
        timer_ID: Identifier of the timer to reset.

    Returns:
        bool: True if the timer was reset, False if the timer didn't exist.
        
    Raises:
        TypeError: If timer_ID is not hashable.
    """
    global _last_time
    
    if not isinstance(timer_ID, Hashable):
        raise TypeError('timer_ID must be a hashable type')
    
    # Reset the timer by removing its entry (treated as "never used")
    return _last_time.pop(timer_ID, None) is not None

