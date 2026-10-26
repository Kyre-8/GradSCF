"""Check phase memory accounting without requiring a GPU or JAX import."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load_monitor():
    path = Path(__file__).resolve().parents[1] / 'examples/training/neural_xc/gpu_memory.py'
    spec = importlib.util.spec_from_file_location('training_gpu_memory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_samples_include_baseline_and_completed_operation():
    module = load_monitor()
    current = [100]
    monitor = module.MemoryMonitor(lambda: current[0], interval=1.)

    def operation():
        current[0] = 250
        return 'completed'

    result, record = monitor.measure('forward', operation, lambda result: None)
    assert result == 'completed'
    assert record['process_start_bytes'] == 100
    assert record['process_end_bytes'] == 250
    assert record['process_sampled_peak_bytes'] == 250
    assert record['process_peak_increase_bytes'] == 150
    assert record['samples'] >= 2
    assert record['seconds'] >= 0


def test_no_gpu_measurement_is_missing_rather_than_zero():
    module = load_monitor()
    monitor = module.MemoryMonitor(lambda: None)
    _, record = monitor.measure('cpu', lambda: 1, lambda result: None)
    assert record['process_sampled_peak_bytes'] is None
    assert record['process_peak_increase_bytes'] is None


def test_synchronization_precedes_the_final_memory_sample():
    module = load_monitor()
    current = [100]
    monitor = module.MemoryMonitor(lambda: current[0], interval=1.)

    def synchronize(result):
        current[0] = 400

    _, record = monitor.measure('async', lambda: 'pending', synchronize)
    assert record['process_end_bytes'] == 400
    assert record['process_sampled_peak_bytes'] == 400


def test_measurement_errors_are_not_silently_reported_as_success():
    module = load_monitor()
    monitor = module.MemoryMonitor(lambda: 100)
    with pytest.raises(RuntimeError, match='operation failed'):
        monitor.measure('failed', lambda: (_ for _ in ()).throw(
            RuntimeError('operation failed')), lambda result: None)


def test_nvml_reader_counts_only_this_pid_across_device_remapping(monkeypatch):
    module = load_monitor()
    processes = {
        0: [SimpleNamespace(pid=os.getpid()+1, usedGpuMemory=999)],
        1: [SimpleNamespace(pid=os.getpid(), usedGpuMemory=250)],
    }
    nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: processes[handle])
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    assert module.nvml_process_reader()() == 250
