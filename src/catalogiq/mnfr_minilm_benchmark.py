"""Scoped timing and a step budget around the unchanged MiniLM training loop.

Instrumentation delegates every loss, optimizer and scheduler operation. It
stops after the requested optimizer AND scheduler steps, before epoch validation
or model export. No configuration, data selection or neural math is changed.
"""
from contextlib import ExitStack
import math
import statistics
import time
from unittest.mock import patch

import numpy as np


class BenchmarkComplete(Exception):
    """Expected control flow after the last authorized benchmark step."""


def _memory_usage(torch):
    result = {}
    try:
        import psutil
        info = psutil.Process().memory_info()
        result.update(process_rss_bytes=info.rss,
                      process_peak_rss_bytes=getattr(info, "peak_wset", info.rss))
    except (ImportError, OSError):
        result["process_rss_available"] = False
    if torch.cuda.is_available():
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


class MnfrBenchmark:
    def __init__(self, corpus, train_indices, config, requested_steps, full_steps, progress=None, *, torch=None):
        if torch is None:
            import torch
        self.torch, self.corpus, self.config = torch, corpus, config
        self.requested_steps, self.full_steps, self.progress = requested_steps, full_steps, progress
        ordering = torch.randperm(len(train_indices), generator=torch.Generator(device="cpu").manual_seed(config.seed)).numpy()
        shuffled = np.asarray(train_indices)[ordering]
        self.batches = [shuffled[start:start + config.batch_size]
                        for start in range(0, requested_steps * config.batch_size, config.batch_size)]
        self.rows = []
        self.optimizer_steps = 0
        self.optimizer = self.optimizer_hook = None
        self.pending_loss = self.pending_labels = self.used_learning_rate = None
        self.initialization_seconds = self.last_tick = None
        self.memory = {}

    def __enter__(self):
        self.stack = ExitStack()
        loss_forward = self.torch.nn.CrossEntropyLoss.forward
        scheduler_step = self.torch.optim.lr_scheduler.LambdaLR.step

        def observe_loss(criterion, logits, labels):
            if self.last_tick is not None:
                if criterion.weight is not None or criterion.label_smoothing != 0:
                    raise ValueError("Benchmark requires unchanged unweighted cross-entropy")
                expected = self.corpus.label_ids[self.batches[len(self.rows)]]
                actual = labels.detach().cpu().numpy()
                if not np.array_equal(actual, expected) or logits.shape[-1] != len(self.corpus.label_names):
                    raise ValueError("Benchmark labels differ from the approved seeded Mnfr batches")
                self.pending_labels = np.bincount(actual, minlength=len(self.corpus.label_names)).tolist()
            output = loss_forward(criterion, logits, labels)
            if self.last_tick is not None:
                self.pending_loss = float(output.detach())
                if not math.isfinite(self.pending_loss):
                    raise FloatingPointError("Nonfinite benchmark loss")
            return output

        def observe_optimizer(optimizer, args, kwargs):
            if self.last_tick is not None:
                self.optimizer_steps += 1
                self.used_learning_rate = float(optimizer.param_groups[0]["lr"])

        def observe_scheduler(scheduler, *args, **kwargs):
            result = scheduler_step(scheduler, *args, **kwargs)
            if self.optimizer is None:
                self.optimizer = scheduler.optimizer
                self.optimizer_hook = self.optimizer.register_step_post_hook(observe_optimizer)
            if self.last_tick is None:  # Ignore LambdaLR's construction-time initial step.
                return result
            if scheduler.optimizer is not self.optimizer or self.optimizer_steps != len(self.rows) + 1 or self.pending_loss is None:
                raise RuntimeError("Benchmark optimizer/scheduler step alignment failed")
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            ended = time.perf_counter()
            duration = ended - self.last_tick
            self.rows.append({"step": self.optimizer_steps, "duration_seconds": duration,
                "loss": self.pending_loss, "learning_rate_used": self.used_learning_rate,
                "batch_rows": int(len(self.batches[len(self.rows)])),
                "label_counts": self.pending_labels, "labels_match_frozen_mapping": True})
            if self.progress:
                self.progress(f"Benchmark step {self.optimizer_steps}/{self.requested_steps}: "
                              f"loss {self.pending_loss:.6f}, {duration:.3f}s")
            self.pending_loss = None
            self.last_tick = time.perf_counter()  # Exclude instrumentation/logging from the next interval.
            if self.optimizer_steps == self.requested_steps:
                self.memory["after_training_steps"] = _memory_usage(self.torch)
                raise BenchmarkComplete()
            return result

        self.stack.enter_context(patch.object(self.torch.nn.CrossEntropyLoss, "forward", observe_loss))
        self.stack.enter_context(patch.object(self.torch.optim.lr_scheduler.LambdaLR, "step", observe_scheduler))
        self.memory["before_model_initialization"] = _memory_usage(self.torch)
        self.started = time.perf_counter()
        return self

    def mark_initialized(self):
        if self.initialization_seconds is not None:
            raise RuntimeError("Benchmark received multiple model initialization signals")
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.initialization_seconds = time.perf_counter() - self.started
        self.memory["model_ready"] = _memory_usage(self.torch)
        self.last_tick = time.perf_counter()

    def __exit__(self, *exc):
        if self.optimizer_hook is not None:
            self.optimizer_hook.remove()
        return self.stack.__exit__(*exc)

    def report(self):
        durations = [row["duration_seconds"] for row in self.rows]
        average = statistics.mean(durations) if durations else None
        return {"requested_steps": self.requested_steps, "completed_optimizer_steps": self.optimizer_steps,
            "completed_scheduler_steps": len(self.rows), "full_schedule_optimizer_steps": self.full_steps,
            "trainer_initialization_seconds": self.initialization_seconds,
            "mean_training_step_seconds": average,
            "median_training_step_seconds": statistics.median(durations) if durations else None,
            "first_training_step_seconds": durations[0] if durations else None,
            "mean_steps_2_onward_seconds": statistics.mean(durations[1:]) if len(durations) > 1 else None,
            "measured_training_step_seconds": sum(durations),
            "estimated_full_training_step_seconds": average * self.full_steps if average is not None else None,
            "estimated_initialization_plus_full_training_seconds": self.initialization_seconds + average * self.full_steps
                if average is not None and self.initialization_seconds is not None else None,
            "timing_scope": "Initialization: legacy trainer entry to model-ready callback; steps: model-ready/previous scheduler completion through next optimizer and scheduler completion, including batch preparation",
            "estimate_excludes": ["input preparation/tokenization", "epoch validation passes", "checkpoint/export time"],
            "all_losses_finite": bool(self.rows) and all(math.isfinite(row["loss"]) for row in self.rows),
            "all_batch_labels_verified": bool(self.rows) and all(row["labels_match_frozen_mapping"] for row in self.rows),
            "class_weighting_enabled": False, "memory": self.memory,
            "memory_scope": "Process peak RSS includes input preparation; CUDA figures are allocator bytes if available",
            "steps": self.rows}
