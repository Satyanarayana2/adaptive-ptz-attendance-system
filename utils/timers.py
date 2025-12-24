# utils/timers.py

import time

class CooldownTimer:
    """
    Simple timer used to avoid repeated actions too quickly.
    """

    def __init__(self, cooldown_seconds=5):
        self.cooldown = cooldown_seconds
        self.last = 0

    def ready(self):
        """
        Returns True if action can be triggered again.
        """
        now = time.time()
        return (now - self.last) >= self.cooldown

    def trigger(self):
        """
        Mark the last action time.
        """
        self.last = time.time()
