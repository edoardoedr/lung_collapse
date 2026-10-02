"""Run control of the fit: time budget, stop reasons, history, checkpoints, watchdog."""

import json
import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager

import numpy as np

log = logging.getLogger(__name__)


class Stop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Tracker:
    """Counts forward solves, keeps the best state and checkpoints it on improvement."""

    def __init__(self, cfg, outdir):
        self.t0 = time.time()
        self.budget = cfg.time_budget_min * 60.0
        self.target = cfg.target_error_mm
        self.level_cap = cfg.level_max_evals
        self.outdir = outdir
        self.best_err, self.best = np.inf, None
        self.level_evals, self.level_best_err, self.level_best_r = 0, np.inf, None
        self.n_eval = self.n_fail = 0
        self.solve_s = self.level_solve_s = 0.0
        self.last_ckpt = 0.0
        self.stop_requested = None
        self.hist = open(outdir / "history.csv", "w")
        self.hist.write("eval,level,K,elapsed_min,mean_err_mm,rms_mm,nu,ok,solve_s\n")

    def elapsed(self):
        return time.time() - self.t0

    def check(self):
        if self.stop_requested:
            raise Stop(self.stop_requested)
        if self.elapsed() > self.budget:
            raise Stop("time budget")
        if self.level_evals >= self.level_cap:
            raise Stop("level eval cap")

    def new_level(self):
        self.level_evals, self.level_best_err, self.level_best_r = 0, np.inf, None
        self.level_solve_s = 0.0

    def log(self, level, K, err, rms, nu, ok, solve_s):
        self.n_eval += 1
        self.level_evals += 1
        self.solve_s += solve_s
        self.level_solve_s += solve_s
        self.hist.write("%d,%d,%d,%.3f,%.5f,%.5f,%.4f,%d,%.3f\n"
                        % (self.n_eval, level, K, self.elapsed() / 60, err, rms, nu, ok, solve_s))
        if self.n_eval % 10 == 0:
            self.hist.flush()

    def improve(self, err, r, state):
        if err < self.level_best_err:
            self.level_best_err, self.level_best_r = err, r.copy()
        if err < self.best_err:
            self.best_err, self.best = err, state
            if time.time() - self.last_ckpt > 15:
                self.checkpoint()
        if err <= self.target:
            self.checkpoint()
            raise Stop("target error reached")

    def checkpoint(self):
        """best_state.npz + best_params.json, so a killed run keeps its best solution."""
        b = self.best
        if b is None:
            return
        arrays = dict(surface_displacement=b["Us"], q=b["q"], nu=b["nu"], tri_labels=b["tri_labels"])
        if isinstance(b["solver_state"], np.ndarray):
            arrays["solver_state"] = b["solver_state"]
        np.savez(self.outdir / "best_state.npz", **arrays)
        (self.outdir / "best_params.json").write_text(json.dumps(
            {k: b[k] for k in ("err", "rms", "K", "level", "nu", "pressures_Pa", "E_Pa", "elapsed_min")},
            indent=2))
        self.last_ckpt = time.time()

    def close(self):
        self.hist.close()


@contextmanager
def guarded(tracker, hard_seconds):
    """Ctrl+C / SIGTERM end the fit cleanly (best result exported); a separate watchdog process
    sends SIGTERM at hard_seconds and SIGKILL 60 s later, also through GIL-blocking solver calls."""
    pid = os.getpid()
    wd = subprocess.Popen(["sh", "-c", "sleep %d; kill -TERM %d 2>/dev/null; sleep 60; kill -KILL %d "
                           "2>/dev/null" % (hard_seconds, pid, pid)], start_new_session=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_signal(signum, frame):
        tracker.stop_requested = "signal %d" % signum

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        try:
            os.killpg(wd.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wd.wait()
