#!/usr/bin/env python3
"""Extract HandNet MAT frames in one archive pass without overwriting originals."""
import argparse
import ctypes as C
import ctypes.util
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def extract(archive, output, max_files=None):
    library = ctypes.util.find_library("archive")
    if not library:
        raise RuntimeError("System libarchive is required")
    lib = C.CDLL(library)
    for name, restype, argtypes in [
        ("archive_read_new", C.c_void_p, []),
        ("archive_read_support_filter_all", C.c_int, [C.c_void_p]),
        ("archive_read_support_format_all", C.c_int, [C.c_void_p]),
        ("archive_read_open_filename", C.c_int, [C.c_void_p, C.c_char_p, C.c_size_t]),
        ("archive_read_next_header", C.c_int, [C.c_void_p, C.POINTER(C.c_void_p)]),
        ("archive_entry_pathname", C.c_char_p, [C.c_void_p]),
        ("archive_entry_filetype", C.c_uint, [C.c_void_p]),
        ("archive_read_data", C.c_ssize_t, [C.c_void_p, C.c_void_p, C.c_size_t]),
        ("archive_read_data_skip", C.c_int, [C.c_void_p]),
        ("archive_error_string", C.c_char_p, [C.c_void_p]),
        ("archive_read_free", C.c_int, [C.c_void_p]),
    ]:
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes
    reader = lib.archive_read_new()
    entry, buffer = C.c_void_p(), C.create_string_buffer(1024 * 1024)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "extraction.json"
    if ledger.exists():
        previous = json.loads(ledger.read_text())
        if previous.get("status") == "complete":
            raise RuntimeError("Extraction already complete; reuse it rather than overwrite")
    state = {"archive": str(archive), "output": str(output), "status": "running",
             "started_at": time.time(), "files": 0, "bytes": 0}
    manifest = output / "frames.jsonl"
    if manifest.exists() and manifest.stat().st_size:
        raise RuntimeError("Partial extraction exists; preserve it and inspect before recovery")

    def check(result):
        if result < 0:
            message = lib.archive_error_string(reader)
            raise RuntimeError(message.decode(errors="replace") if message else str(result))

    try:
        check(lib.archive_read_support_filter_all(reader))
        check(lib.archive_read_support_format_all(reader))
        check(lib.archive_read_open_filename(reader, os.fsencode(archive), 1024 * 1024))
        atomic_json(ledger, state)
        with manifest.open("w") as records:
            while True:
                code = lib.archive_read_next_header(reader, C.byref(entry))
                if code == 1:  # ARCHIVE_EOF
                    break
                check(code)
                name = lib.archive_entry_pathname(entry).decode("utf-8")
                rel = PurePosixPath(name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise RuntimeError("Unsafe archive path: " + name)
                if lib.archive_entry_filetype(entry) != 0o100000 or not rel.name.startswith("Data_") or rel.suffix != ".mat":
                    check(lib.archive_read_data_skip(reader))
                    continue
                target = output.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with target.open("xb") as handle:
                    while True:
                        count = lib.archive_read_data(reader, buffer, len(buffer))
                        check(count)
                        if count == 0:
                            break
                        chunk = buffer.raw[:count]
                        handle.write(chunk)
                        digest.update(chunk)
                        size += count
                records.write(json.dumps({"path": rel.as_posix(), "bytes": size,
                                          "sha256": digest.hexdigest()}) + "\n")
                state["files"] += 1
                state["bytes"] += size
                if state["files"] % 1000 == 0:
                    state["updated_at"] = time.time()
                    atomic_json(ledger, state)
                    records.flush()
                    print(json.dumps(state), flush=True)
                if max_files and state["files"] >= max_files:
                    break
        state.update(status="complete", updated_at=time.time(), limited=bool(max_files))
        atomic_json(ledger, state)
        print(json.dumps(state), flush=True)
        return state
    except BaseException as error:
        state.update(status="failed", error=repr(error), updated_at=time.time())
        atomic_json(ledger, state)
        raise
    finally:
        lib.archive_read_free(reader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    extract(args.archive.resolve(), args.output.resolve(), args.max_files)
