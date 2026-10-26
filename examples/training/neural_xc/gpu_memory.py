"""GPU memory sampling for the accompanying H2 neural-XC training example.

NVML samples are observed residency, including CUDA/library/allocator overhead.
They are lower bounds on the true peak; a 10 ms poll can miss shorter spikes.
JAX allocator statistics should be recorded separately by the caller.
"""
import os
from threading import Event, Thread
from time import perf_counter


def nvml_process_reader():
    """Return a PID-based reader, independent of CUDA_VISIBLE_DEVICES ordering."""
    import pynvml

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
               for i in range(pynvml.nvmlDeviceGetCount())]
    pid = os.getpid()

    def read():
        total = 0
        for handle in handles:
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if process.pid == pid:
                    memory = process.usedGpuMemory
                    # pynvml normalizes NVML's unavailable uint64 sentinel to None.
                    if memory is None:
                        raise RuntimeError('NVML process memory is unavailable.')
                    total += int(memory)
        return total

    return read


class MemoryMonitor:
    """Measure one phase using an injectable process-memory reader."""

    def __init__(self, reader, *, interval=.01):
        if interval <= 0:
            raise ValueError('Memory sampling interval must be positive.')
        self.reader = reader
        self.interval = interval

    def measure(self, name, operation, synchronize):
        samples, errors = [self.reader()], []
        stop = Event()

        def poll():
            while not stop.wait(self.interval):
                try:
                    samples.append(self.reader())
                except Exception as error:
                    errors.append(error)
                    return

        thread = Thread(target=poll, daemon=True)
        thread.start()
        started = perf_counter()
        try:
            result = operation()
            synchronize(result)
        finally:
            elapsed = perf_counter() - started
            stop.set()
            thread.join()
        if errors:
            raise errors[0]
        samples.append(self.reader())
        valid = [sample for sample in samples if sample is not None]
        peak = max(valid) if valid else None
        record = dict(phase=name, seconds=elapsed, samples=len(valid),
            sample_interval_seconds=self.interval,
            process_start_bytes=samples[0], process_end_bytes=samples[-1],
            process_sampled_peak_bytes=peak,
            process_peak_increase_bytes=(None if peak is None or samples[0] is None
                                         else peak - samples[0]))
        return result, record
