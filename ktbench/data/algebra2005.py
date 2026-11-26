"""Timestamp-safe HiTSKT views for Algebra's simultaneous transactions."""
from __future__ import annotations


import numpy as np

from ktbench.batching import ExampleShape, Session, SessionExample
from ktbench.data.session_store import RollingSessionDataset, SPLIT_NAMES


class AlgebraSessionTargets(RollingSessionDataset):
    """Predict all events, including each student's first session.

    For each later row at the same timestamp, score a second causal prefix
    containing the strictly earlier session events followed by that row.
    The main example can use all tied outcomes once the clock advances.
    """

    def __init__(self, store, split, *, history_sessions=15):
        super().__init__(store, split, history_sessions=history_sessions)
        self.targets = np.flatnonzero(store.session_split == SPLIT_NAMES[split])
        self.events = store.events_in_split(self.split)
        self.target_count = len(self.events)
        tied_events = self.events[store.history_stop(self.events) != self.events]
        self.tied_sessions = np.searchsorted(store.session_offsets, tied_events, side="right") - 1
        self.tied_events = tied_events

    def __len__(self):
        return len(self.targets) + len(self.tied_events)

    def _target(self, index):
        if index < len(self.targets):
            return int(self.targets[index]), None
        index -= len(self.targets)
        return int(self.tied_sessions[index]), int(self.tied_events[index])

    def _indices(self, session, extra):
        start, stop = self.store.session_offsets[session:session + 2]
        if extra is None:
            return np.arange(start, stop), 0
        return np.r_[np.arange(start, self.store.history_stop(extra)), extra], int(
            self.store.history_stop(extra) - start)

    def __getitem__(self, index):
        session, extra = self._target(index)
        events, metric_start = self._indices(session, extra)
        start, stop = self._history_bounds(session)
        history = [self.store.session(previous) for previous in range(start, stop)]
        mask = (self.store.history_stop(events) == events) if extra is None else None
        return SessionExample(
            history, Session(self.store.question[events], self.store.skill[events],
                             self.store.correct[events]), self.split, metric_start=metric_start,
            event_start=int(events[0]), metric_positions=mask,
            event_ids=events if extra is not None else None,
        )

    def shape(self, index):
        session, extra = self._target(index)
        start, stop = self._history_bounds(session)
        offsets = self.store.session_offsets
        target_length = (self.store.session_length(session) if extra is None
                         else int(self.store.history_stop(extra) - offsets[session] + 1))
        return ExampleShape(tuple(np.diff(offsets[start:stop + 1]).tolist()), target_length)
