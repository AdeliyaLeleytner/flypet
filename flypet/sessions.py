"""Bounded temporary visitor memories; no second connectome per visitor."""

from collections import OrderedDict, deque
from threading import Lock
import secrets
import time


class SessionStore:
    def __init__(self, capacity=32, ttl=3600):
        self.capacity, self.ttl = capacity, ttl
        self.items, self.created = OrderedDict(), deque()
        self.lock = Lock()

    def create(self):
        with self.lock:
            now = time.monotonic()
            while self.created and now - self.created[0] > 60:
                self.created.popleft()
            if len(self.created) >= 20:
                raise RuntimeError("Session creation rate exceeded")
            self.created.append(now)
            token = secrets.token_urlsafe(32)
            self.items[token] = {
                "touched": now,
                "memory": None,
                "log": [],
                "history": [],
                "counter": 0,
                "state": {
                    "satiety": 0.3,
                    "born": "session",
                    "last_tick": time.time(),
                    "events": [],
                    "n_stimuli": 0,
                    "name": "Fly",
                },
            }
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)
            return token

    def contains(self, token):
        with self.lock:
            entry = self.items.get(token)
            if entry is None:
                return False
            if time.monotonic() - entry["touched"] > self.ttl:
                del self.items[token]
                return False
            return True

    def get(self, token):
        with self.lock:
            entry = self.items.get(token)
            if entry is None or time.monotonic() - entry["touched"] > self.ttl:
                raise ValueError("Your session expired. Reconnect to start again.")
            entry["touched"] = time.monotonic()
            self.items.move_to_end(token)
            return entry
