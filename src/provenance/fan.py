"""Filesystem-wide write watcher over raw fanotify (ctypes, no dependencies).

One FAN_MARK_FILESYSTEM mark covers every directory on the disk, so a multi-terabyte
tree costs one mark, not millions of inotify watches. Events are reported by file
handle (FAN_REPORT_DFID_NAME) and resolved back to paths with open_by_handle_at.

Requires root (CAP_SYS_ADMIN for the mark, CAP_DAC_READ_SEARCH for open_by_handle_at)
and a filesystem that supports file handles (ext4, xfs, btrfs; not FUSE/ntfs-3g).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import select
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

FAN_MODIFY = 0x00000002
FAN_CLOSE_WRITE = 0x00000008
FAN_MOVED_FROM = 0x00000040
FAN_MOVED_TO = 0x00000080
FAN_CREATE = 0x00000100
FAN_DELETE = 0x00000200
FAN_ONDIR = 0x40000000

FAN_CLOEXEC = 0x00000001
FAN_NONBLOCK = 0x00000002
FAN_CLASS_NOTIF = 0x00000000
FAN_UNLIMITED_QUEUE = 0x00000010
FAN_REPORT_DIR_FID = 0x00000400
FAN_REPORT_NAME = 0x00000800
FAN_REPORT_DFID_NAME = FAN_REPORT_DIR_FID | FAN_REPORT_NAME

FAN_MARK_ADD = 0x00000001
FAN_MARK_FILESYSTEM = 0x00000100

FAN_EVENT_INFO_TYPE_DFID_NAME = 2
FAN_EVENT_INFO_TYPE_DFID = 3

FAN_Q_OVERFLOW = 0x00004000

O_PATH = 0o10000000
AT_FDCWD = -100

WATCH_MASK = FAN_CREATE | FAN_MOVED_TO | FAN_CLOSE_WRITE | FAN_DELETE | FAN_MOVED_FROM | FAN_ONDIR

_META = struct.Struct("=IBBHQii")  # event_len, vers, reserved, metadata_len, mask, fd, pid
_INFO_HDR = struct.Struct("=BBH")
_FSID = 8
_HANDLE_HDR = struct.Struct("=Ii")

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.fanotify_init.argtypes = [ctypes.c_uint, ctypes.c_uint]
_libc.fanotify_init.restype = ctypes.c_int
_libc.fanotify_mark.argtypes = [
    ctypes.c_int, ctypes.c_uint, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p,
]
_libc.fanotify_mark.restype = ctypes.c_int
_libc.open_by_handle_at.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_libc.open_by_handle_at.restype = ctypes.c_int


class FanotifyError(OSError):
    pass


@dataclass(frozen=True)
class FanEvent:
    path: str  # absolute; may be "" if the directory vanished before we resolved it
    mask: int
    pid: int

    @property
    def is_dir(self) -> bool:
        return bool(self.mask & FAN_ONDIR)

    @property
    def is_removal(self) -> bool:
        return bool(self.mask & (FAN_DELETE | FAN_MOVED_FROM))


class Watcher:
    """Watch every write on the filesystem holding `root`."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        flags = FAN_CLOEXEC | FAN_NONBLOCK | FAN_CLASS_NOTIF | FAN_REPORT_DFID_NAME
        flags |= FAN_UNLIMITED_QUEUE
        fd = _libc.fanotify_init(flags, os.O_RDONLY | os.O_LARGEFILE)
        if fd < 0:
            e = ctypes.get_errno()
            raise FanotifyError(e, f"fanotify_init: {os.strerror(e)} (needs root, kernel >= 5.9)")
        self.fd = fd
        rc = _libc.fanotify_mark(
            fd, FAN_MARK_ADD | FAN_MARK_FILESYSTEM, WATCH_MASK, AT_FDCWD, self.root.encode()
        )
        if rc < 0:
            e = ctypes.get_errno()
            os.close(fd)
            raise FanotifyError(e, f"fanotify_mark {self.root}: {os.strerror(e)}")
        # open_by_handle_at needs any fd on the same filesystem.
        self.mount_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.overflowed = False

    def close(self) -> None:
        for fd in (self.fd, self.mount_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _resolve(self, handle: bytes) -> str:
        buf = ctypes.create_string_buffer(handle, len(handle))
        fd = _libc.open_by_handle_at(self.mount_fd, buf, O_PATH)
        if fd < 0:
            return ""
        try:
            return os.readlink(f"/proc/self/fd/{fd}")
        finally:
            os.close(fd)

    def _parse(self, data: bytes) -> Iterator[FanEvent]:
        off = 0
        while off + _META.size <= len(data):
            event_len, _vers, _res, meta_len, mask, fd, pid = _META.unpack_from(data, off)
            if event_len < _META.size:
                break
            if fd >= 0:
                os.close(fd)
            if mask & FAN_Q_OVERFLOW:
                self.overflowed = True
            info = off + meta_len
            end = off + event_len
            while info + _INFO_HDR.size <= end:
                itype, _pad, ilen = _INFO_HDR.unpack_from(data, info)
                if ilen == 0:
                    break
                if itype in (FAN_EVENT_INFO_TYPE_DFID_NAME, FAN_EVENT_INFO_TYPE_DFID):
                    h = info + _INFO_HDR.size + _FSID
                    hbytes, _htype = _HANDLE_HDR.unpack_from(data, h)
                    handle = data[h : h + _HANDLE_HDR.size + hbytes]
                    name = b""
                    if itype == FAN_EVENT_INFO_TYPE_DFID_NAME:
                        n0 = h + _HANDLE_HDR.size + hbytes
                        raw = data[n0 : info + ilen]
                        name = raw.split(b"\0", 1)[0]
                    d = self._resolve(handle)
                    if d and name and name != b".":
                        path = os.path.join(d, os.fsdecode(name))
                    else:
                        path = d
                    yield FanEvent(path=path, mask=mask, pid=pid)
                    break
                info += ilen
            off = end

    def read(self, timeout: float | None = None) -> list[FanEvent]:
        """Return the events available within `timeout` seconds (possibly none)."""
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return []
        out: list[FanEvent] = []
        while True:
            try:
                data = os.read(self.fd, 1 << 16)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                raise
            if not data:
                break
            out.extend(self._parse(data))
        return out


FAN_OPEN = 0x00000020
FAN_OPEN_EXEC = 0x00001000
FAN_MARK_MOUNT = 0x00000010
_META_FD = _META  # same fanotify_event_metadata layout; here fd is a real open fd


class OpenWatcher:
    """Every open (and exec) on one mount, as (path, pid). For "is this still used?".

    Mount marks with fd reporting need no file handles, so this works where `Watcher`
    cannot (ntfs3, FUSE), and on noatime mounts where atime says nothing. It sees opens
    only while it runs; callers track coverage themselves.
    """

    def __init__(self, mountpoint: str) -> None:
        self.mountpoint = mountpoint
        fd = _libc.fanotify_init(FAN_CLOEXEC | FAN_NONBLOCK | FAN_CLASS_NOTIF
                                 | FAN_UNLIMITED_QUEUE, os.O_RDONLY | os.O_LARGEFILE)
        if fd < 0:
            e = ctypes.get_errno()
            raise FanotifyError(e, f"fanotify_init: {os.strerror(e)} (needs root)")
        self.fd = fd
        rc = _libc.fanotify_mark(fd, FAN_MARK_ADD | FAN_MARK_MOUNT, FAN_OPEN | FAN_OPEN_EXEC,
                                 AT_FDCWD, mountpoint.encode())
        if rc < 0:
            e = ctypes.get_errno()
            os.close(fd)
            raise FanotifyError(e, f"fanotify_mark (mount) {mountpoint}: {os.strerror(e)}")
        self.overflowed = False

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass

    def read(self, timeout: float | None = None) -> list[tuple[str, int]]:
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return []
        out: list[tuple[str, int]] = []
        while True:
            try:
                data = os.read(self.fd, 1 << 16)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                raise
            if not data:
                break
            off = 0
            while off + _META_FD.size <= len(data):
                event_len, _v, _r, _ml, mask, efd, pid = _META_FD.unpack_from(data, off)
                if event_len < _META_FD.size:
                    break
                if mask & FAN_Q_OVERFLOW:
                    self.overflowed = True
                if efd >= 0:
                    try:
                        out.append((os.readlink(f"/proc/self/fd/{efd}"), pid))
                    except OSError:
                        pass
                    finally:
                        os.close(efd)
                off += event_len
        return out
