"""Windows worker job limits. Standard library only; never initializes ML/CUDA."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC_LIMITS(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class EXTENDED_LIMITS(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", BASIC_LIMITS), ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class PROCESS_MEMORY(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class PROCESS_IDS(ctypes.Structure):
    _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD),
                ("NumberOfProcessIdsInList", wintypes.DWORD),
                ("ProcessIdList", ctypes.c_size_t * 2)]


class THREAD_ENTRY(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD)]


class Job:
    def __init__(self, memory_bytes):
        if os.name != "nt":
            raise RuntimeError("This capped experiment requires Windows job objects")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self.kernel.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREAD_ENTRY)]
        self.kernel.Thread32First.restype = wintypes.BOOL
        self.kernel.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREAD_ENTRY)]
        self.kernel.Thread32Next.restype = wintypes.BOOL
        self.kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenThread.restype = wintypes.HANDLE
        self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.ResumeThread.restype = wintypes.DWORD
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetExitCodeProcess.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = EXTENDED_LIMITS()
        # Launcher plus interpreter share one aggregate cap; kill on handle close.
        limits.BasicLimitInformation.LimitFlags = 0x100 | 0x200 | 0x8 | 0x2000
        limits.BasicLimitInformation.ActiveProcessLimit = 2
        limits.ProcessMemoryLimit = memory_bytes
        limits.JobMemoryLimit = memory_bytes
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume(self, process):
        """Resume the sole thread of a CREATE_SUSPENDED launcher after assignment."""
        if process.pid not in self.process_ids():
            raise RuntimeError("Suspended launcher must be in the capped job before resume")
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = THREAD_ENTRY()
            entry.dwSize = ctypes.sizeof(entry)
            found = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            threads = []
            while found:
                if entry.th32OwnerProcessID == process.pid:
                    threads.append(entry.th32ThreadID)
                found = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            if len(threads) != 1:
                raise RuntimeError("Expected exactly one suspended launcher thread")
            thread = self.kernel.OpenThread(0x2, False, threads[0])
            if not thread:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if self.kernel.ResumeThread(thread) != 1:
                    raise RuntimeError("Launcher was not suspended exactly once")
            finally:
                self.kernel.CloseHandle(thread)
        finally:
            self.kernel.CloseHandle(snapshot)

    def process_ids(self):
        # Windows returns a variable-size list. Reserve slack and honor both its
        # required byte count and assigned-process count on ERROR_MORE_DATA.
        offset = PROCESS_IDS.ProcessIdList.offset
        pointer_bytes = ctypes.sizeof(ctypes.c_size_t)
        capacity = 8
        while capacity <= 65536:
            buffer = ctypes.create_string_buffer(offset + capacity * pointer_bytes)
            returned = wintypes.DWORD()
            success = self.kernel.QueryInformationJobObject(
                self.handle, 3, ctypes.byref(buffer), ctypes.sizeof(buffer), ctypes.byref(returned))
            entries = PROCESS_IDS.from_buffer(buffer)
            if success:
                count = entries.NumberOfProcessIdsInList
                if count > capacity:
                    raise RuntimeError("Windows returned an invalid job process-list size")
                values = (ctypes.c_size_t * count).from_buffer(buffer, offset)
                return list(values)
            error = ctypes.get_last_error()
            if error != 234:
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, entries.NumberOfAssignedProcesses + 1,
                           (returned.value + pointer_bytes - 1) // pointer_bytes)
        raise RuntimeError("Job process list exceeded the bounded query buffer")

    def usage(self):
        """Sum working set/private memory across launcher and interpreter."""
        totals = {"working_set_bytes": 0, "private_bytes": 0}
        for pid in self.process_ids():
            handle = self.kernel.OpenProcess(0x400 | 0x10, False, pid)
            if not handle:
                error = ctypes.get_last_error()
                if error == 87:  # Process exited between enumeration and OpenProcess.
                    continue
                raise ctypes.WinError(error)
            try:
                try:
                    counters = _memory_for_handle(handle)
                except OSError:
                    code = wintypes.DWORD()
                    if self.kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:
                        continue
                    raise
                totals["working_set_bytes"] += counters["working_set_bytes"]
                totals["private_bytes"] += counters["private_bytes"]
            finally:
                self.kernel.CloseHandle(handle)
        return totals

    def query(self):
        limits = EXTENDED_LIMITS()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "process_commit_limit_bytes": limits.ProcessMemoryLimit,
            "job_commit_limit_bytes": limits.JobMemoryLimit,
            "peak_process_commit_bytes": limits.PeakProcessMemoryUsed,
            "peak_job_commit_bytes": limits.PeakJobMemoryUsed,
            "active_process_limit": limits.BasicLimitInformation.ActiveProcessLimit,
        }

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 124):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _memory_for_handle(handle):
    api = ctypes.WinDLL("psapi", use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = PROCESS_MEMORY()
    counters.cb = ctypes.sizeof(counters)
    if not api.GetProcessMemoryInfo(handle, ctypes.byref(counters), ctypes.sizeof(counters)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"working_set_bytes": counters.WorkingSetSize,
            "peak_working_set_bytes": counters.PeakWorkingSetSize,
            "private_bytes": counters.PrivateUsage}


def process_memory(process):
    return _memory_for_handle(wintypes.HANDLE(int(process._handle)))


class ACCOUNTING(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in (
        'TotalUserTime','TotalKernelTime','ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime')]
    _fields_ += [(name, wintypes.DWORD) for name in (
        'TotalPageFaultCount','TotalProcesses','ActiveProcesses','TotalTerminatedProcesses')]


class BoundedJob(Job):
    def __init__(self, memory_bytes, cpu_seconds, threads):
        super().__init__(memory_bytes)
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p]
        self.kernel.GetProcessAffinityMask.restype = wintypes.BOOL
        available, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not self.kernel.GetProcessAffinityMask(self.kernel.GetCurrentProcess(), ctypes.byref(available), ctypes.byref(system)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        cpus = [bit for bit in range(ctypes.sizeof(ctypes.c_size_t) * 8) if available.value & (1 << bit)]
        if len(cpus) < threads:
            self.close()
            raise RuntimeError("Fewer approved CPUs are available than expected; do not start")
        selected = [cpus[(i * len(cpus)) // threads] for i in range(threads)]
        self.affinity = sum(1 << bit for bit in selected)
        limits = EXTENDED_LIMITS()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        limits.BasicLimitInformation.LimitFlags |= 0x10 | 0x4  # AFFINITY and JOB_TIME
        limits.BasicLimitInformation.Affinity = self.affinity
        limits.BasicLimitInformation.PerJobUserTimeLimit = int(cpu_seconds * 10_000_000)
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def accounting(self):
        value = ACCOUNTING()
        if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"user_seconds": value.TotalUserTime / 10_000_000,
                "kernel_seconds": value.TotalKernelTime / 10_000_000,
                "cpu_seconds": (value.TotalUserTime + value.TotalKernelTime) / 10_000_000,
                "total_processes": value.TotalProcesses,
                "active_processes": value.ActiveProcesses,
                "terminated_for_limits": value.TotalTerminatedProcesses}

    def enforcement(self):
        value = EXTENDED_LIMITS()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {**self.query(), "affinity_mask": value.BasicLimitInformation.Affinity,
                "allowed_cpu_count": int(value.BasicLimitInformation.Affinity).bit_count(),
                "kernel_user_cpu_limit_seconds": value.BasicLimitInformation.PerJobUserTimeLimit / 10_000_000,
                "limit_flags": value.BasicLimitInformation.LimitFlags}
