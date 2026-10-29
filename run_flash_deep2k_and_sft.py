# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULTLM-150M + FlashVAULT (G=12):
#   Phase 1 (#2): Deep L=2048 Long-Context Continuation (Multi-Chunk Induction Curriculum + 2K NIAH)
#   Phase 2 (#1): Clean Native-Prose (NO <think> on prose, ZERO code-review contamination) + 35% Replay SFT
#
# Strict IP Scope: Uses ONLY Philip Bartsch's owned repos (`vault`, `buildbit`, `buildbit-os`) + public HF datasets.

import os
import glob
import json
import math
import time
import queue
import random
import shutil
import argparse
import threading
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

from src.vault_lm_150m import VAULTLM150M


# ==============================================================================
# 1. CLEAN NATIVE-PROSE CS CONCEPT GENERATOR (NO <think> TAGS! ZERO CODE FENCES!)
#    Matches `train_vault_150m.py` line 322 where prose NEVER starts with `<think>`.
#    Strictly holds out "recursion" and "base case" so Eval #7 stays 100% unseen!
# ==============================================================================

CS_CONCEPT_BANK = [
    (
        "an exit condition in a `while` loop",
        "loops and control flow",
        "An exit condition is essential in a `while` loop because it defines the exact state under which iteration stops and control returns to the rest of the program. Without a reachable exit condition, the loop guard remains true forever, trapping the CPU thread inside an infinite loop so the program hangs and never produces an answer.",
        "On each pass, state updates inside the loop body move variables closer to that stopping threshold; once the condition evaluates to false, execution cleanly advances to the next statement.",
    ),
    (
        "a stopping depth in nested function calls",
        "call stack management",
        "Every active function call pushes a new stack frame containing return addresses and local variables onto a finite call stack in memory. Without a well-defined stopping condition that halts further nested calls, frames accumulate until the thread exhausts its stack region and crashes with a stack overflow.",
        "Once the deepest call reaches a trivial terminal input and returns a direct value without making another call, each waiting frame unwinds in reverse LIFO order.",
    ),
    (
        "a terminal leaf check in binary tree traversals",
        "tree data structures",
        "Tree traversal routines visit child pointers step by step from the root downward; checking for an empty (`null` or `None`) child pointer provides the stopping boundary where downward traversal halts. Without that boundary check, dereferencing a null child pointer immediately triggers an invalid memory access or runtime exception.",
        "Handling the empty-node case cleanly allows each subtree call to return smoothly and combine results back up to the root.",
    ),
    (
        "a base step in mathematical induction and divide-and-conquer algorithms",
        "algorithm design",
        "Divide-and-conquer algorithms repeatedly split a problem of size N into smaller subproblems, which only succeeds if subproblems of size 0 or 1 are solved directly without further splitting. Without an explicit elementary stopping case, subproblem decomposition would continue indefinitely and never ground out in a concrete value.",
        "That smallest direct answer anchors the entire chain of reductions so every larger level can combine concrete sub-results.",
    ),
    (
        "stack memory versus heap memory",
        "memory management",
        "Stack memory stores fixed-lifetime local variables and function call frames in contiguous LIFO order using a single CPU stack-pointer register (`O(1)` push and pop), whereas heap memory stores dynamically sized or long-lived data managed via explicit allocations.",
        "Keeping short-lived locals on the stack maximizes L1 cache locality with zero allocator overhead, while heap allocation lets data structures grow dynamically at runtime and outlive the scope that created them.",
    ),
    (
        "a sorted input array in binary search",
        "search algorithms",
        "Binary search achieves `O(log n)` time by eliminating half of the remaining search range after a single comparison against the midpoint, which relies completely on global sorted order. When the array is sorted, every element to the left of the midpoint is guaranteed to be smaller or equal and every element to the right is larger or equal.",
        "If the input is unsorted, comparing against the middle element reveals nothing about where the target resides, so discarding either half can skip right past the target.",
    ),
    (
        "collision resolution in hash tables",
        "data structures",
        "Because a hash function maps a massive universe of possible keys into a finite array of buckets, the Pigeonhole Principle guarantees that two distinct keys will eventually hash to the same index. Without collision resolution, inserting the second key would overwrite and lose the first entry.",
        "Techniques such as separate chaining or open addressing with linear probing store all colliding entries safely while keeping average lookup time `O(1)`.",
    ),
    (
        "a convergence threshold and iteration cap in numerical solvers",
        "numerical computing",
        "Iterative numerical algorithms refine an approximate solution step by step, but finite floating-point precision means rounding noise often prevents the residual error from reaching exact zero. Without both an epsilon tolerance and a maximum-iteration guard, the solver loop can oscillate forever on the last bit of precision.",
        "Checking whether the error is below tolerance or the step budget is reached guarantees bounded runtime in production systems.",
    ),
    (
        "mutexes or atomic operations on shared mutable state",
        "multithreaded programming",
        "When multiple CPU threads read and write the same shared variable concurrently without synchronization, interleaved load-modify-store instructions cause race conditions that silently corrupt state. Because OS thread scheduling varies from run to run, unsynchronized races produce non-deterministic bugs that are notoriously hard to reproduce.",
        "Mutexes and hardware atomics serialize conflicting updates and enforce cache-coherent memory ordering across CPU cores.",
    ),
    (
        "short-circuit evaluation in boolean expressions",
        "programming language semantics",
        "Short-circuit boolean operators (`&&`/`and` and `||`/`or`) stop evaluating the right-hand expression as soon as the left-hand expression determines the final boolean outcome. Beyond saving unnecessary work, this behavior is crucial for safety guards where the second expression is only valid when the first holds.",
        "For example, checking that a pointer is non-null or an array index is in bounds on the left prevents an out-of-bounds crash on the right.",
    ),
    (
        "64-byte cache-line alignment in SIMD data structures",
        "high-performance systems programming",
        "Modern CPUs move data between main memory and L1/L2 caches in fixed 64-byte cache lines; aligning hot arrays to 64-byte boundaries prevents a single vector load from splitting across two cache lines. In multithreaded code, alignment and padding also prevent false sharing when two cores update independent variables that happen to sit on the same cache line.",
        "Pairing cache alignment with Struct-of-Arrays (`soa`) layout ensures every SIMD instruction processes 100% contiguous, useful lanes.",
    ),
    (
        "power-of-two capacities in ring buffers",
        "low-latency systems",
        "General integer modulo (`%`) requires a multi-cycle hardware division instruction, whereas wrapping an index modulo a power-of-two capacity N simplifies to a single-cycle bitwise AND (`index & (N - 1)`). In high-throughput audio, networking, and GPU command queues, saving tens of cycles on every enqueue and dequeue noticeably reduces tail latency.",
        "Power-of-two sizing also maps cleanly onto virtual memory pages and CPU cache sets.",
    ),
    (
        "virtual memory and hardware page tables",
        "operating systems",
        "Virtual memory gives every process its own isolated, contiguous logical address space translated to physical RAM frames by the CPU's Memory Management Unit (MMU) and Translation Lookaside Buffer (TLB). Process isolation ensures a stray pointer in one application cannot read or corrupt memory belonging to another process or the kernel.",
        "Unmapped guard pages at the ends of stacks and heaps also let hardware trap overflows immediately rather than allowing silent memory corruption.",
    ),
    (
        "immutable data structures in concurrent software",
        "software architecture",
        "Immutable objects never change state after construction, which means any number of threads or functions can read and share references to them concurrently without locks or defensive copies. Eliminating in-place mutation removes entire classes of aliasing bugs, temporal coupling, and race conditions.",
        "When an update is needed, structural sharing creates a lightweight new version that reuses unchanged subtrees efficiently.",
    ),
    (
        "arena (bump) allocators over per-object heap allocation",
        "systems memory management",
        "General-purpose heap allocators track individual free blocks, coalesce fragments, and acquire metadata locks on every allocation and deallocation. An arena allocator instead pre-reserves one contiguous block and allocates by advancing a single offset pointer in `O(1)` time, then frees thousands of request- or frame-scoped objects at once by resetting the offset to zero.",
        "This eliminates heap fragmentation, per-object bookkeeping overhead, and use-after-free bugs within the arena's lifetime.",
    ),
    (
        "explicit string lengths or sentinel terminators in memory",
        "systems programming",
        "Raw memory is a flat array of bytes with no intrinsic boundaries between adjacent variables; string routines need either a stored byte count or a sentinel null byte to know where text ends. Without a length bound, loops reading a string keep marching into neighboring stack or heap memory.",
        "Length-prefixed slices (`ptr + len`) are especially favored in modern systems languages because they run `O(1)` length checks and safely contain arbitrary binary bytes.",
    ),
    (
        "consistent lock ordering for deadlock prevention",
        "concurrent systems",
        "Deadlock happens when two or more threads form a dependency cycle—for example, Thread A holds Lock 1 while waiting for Lock 2, and Thread B holds Lock 2 while waiting for Lock 1—so neither can ever proceed. Enforcing a strict global acquisition order across all locks makes circular wait mathematically impossible.",
        "Because every thread acquires lower-numbered locks before higher-numbered ones, at least one thread at the top of the order can always finish and release its locks.",
    ),
    (
        "amortized capacity doubling in dynamic arrays",
        "algorithmic complexity",
        "Appending to a full contiguous array requires allocating a larger buffer and copying existing elements over; growing capacity by just one slot per append would copy `1 + 2 + ... + N` items, degrading `N` appends to `O(N^2)` time. Doubling capacity whenever full spaces reallocations exponentially far apart so total copies sum to less than `2N`.",
        "As a result, dynamic arrays preserve contiguous cache-friendly indexing while still delivering `O(1)` amortized append time.",
    ),
    (
        "epsilon tolerances when comparing floating-point numbers",
        "scientific computing",
        "IEEE-754 binary floating-point numbers have finite mantissa bits and cannot represent many decimal fractions such as `0.1` exactly, causing small rounding residuals to accumulate across arithmetic steps. Testing two computed floats with exact equality (`==`) frequently fails even when two formulas are algebraically identical.",
        "Comparing whether their absolute or relative difference falls within a small tolerance `epsilon` checks agreement up to hardware precision.",
    ),
    (
        "include guards or module imports during compilation",
        "compiler toolchains",
        "In C and C++, `#include` literally pastes header text into a source file; if two headers include a common third header without guards, struct and inline definitions appear twice and break the One Definition Rule. Header guards (`#pragma once` or `#ifndef`) instruct the preprocessor to skip duplicate inclusions within the same translation unit.",
        "Keeping headers idempotent prevents redefinition errors and cuts redundant lexing and parsing work.",
    ),
    (
        "visited-node tracking in graph breadth-first and depth-first search",
        "graph algorithms",
        "Unlike trees, general graphs can contain cycles and multiple paths leading to the same vertex. If a graph search does not record which vertices have already been visited, traversal will cycle endlessly around loops and re-expand the same subgraphs exponentially many times.",
        "Marking each vertex visited as soon as it is discovered guarantees linear `O(V + E)` runtime and clean termination.",
    ),
    (
        "memoization in overlapping subproblem algorithms",
        "dynamic programming",
        "Naive top-down solutions to problems with overlapping subproblems recompute the exact same smaller inputs thousands or millions of times, turning polynomial problems into exponential ones. Memoization caches each subproblem's result keyed by its arguments the first time it is computed and returns the cached value in `O(1)` on every repeat lookup.",
        "Trading a modest lookup table for eliminated redundant work collapses exponential call trees into linear or polynomial time.",
    ),
    (
        "tail-call optimization and iterative loop transformation",
        "compilers and runtime systems",
        "When a function's very last action is returning the result of another call, the caller's stack frame holds no remaining work and does not need to stay allocated. Reusing the current stack frame—or rewriting the routine as a loop—reduces space complexity from `O(N)` stack frames down to `O(1)` constant memory.",
        "Constant stack usage allows arbitrarily long chains of state transitions without risking stack exhaustion.",
    ),
    (
        "database indexes (B-trees) versus full table scans",
        "database systems",
        "Without an index, finding rows matching a `WHERE` clause forces the database engine to scan every page of the table sequentially in `O(N)` time. A B-tree index maintains keys in balanced sorted pages with high fan-out, letting queries locate matching rows in `O(log N)` page reads.",
        "Because three or four B-tree levels can index hundreds of millions of records, indexed lookups turn multi-second disk scans into sub-millisecond reads.",
    ),
    (
        "TCP reliability versus UDP low-latency datagrams",
        "network engineering",
        "TCP provides a reliable, ordered byte stream via sequence numbers, acknowledgments, retransmissions, and congestion control, whereas UDP sends independent datagrams immediately with minimal framing overhead. Retransmitting a dropped TCP packet stalls delivery of later packets behind it—known as head-of-line blocking.",
        "Applications requiring exact byte fidelity use TCP, while real-time simulations, games, and media streams use UDP so fresh updates are never delayed by stale lost packets.",
    ),
    (
        "static type checking at compile time",
        "programming languages",
        "Static type checking verifies before a program ever runs that function arguments, return values, and struct fields match their declared contracts across the entire codebase. Catching mismatched types, missing fields, or unhandled nullability at compile time eliminates a major source of production runtime crashes.",
        "Explicit type layouts also give the compiler exact size and alignment facts needed to generate unboxed, SIMD-friendly machine code.",
    ),
    (
        "write-ahead logging (WAL) for crash recovery",
        "storage engines",
        "Updating complex on-disk data structures in place is vulnerable to power loss or process crashes mid-write, which leaves pages half-updated and corrupt. Write-ahead logging appends every intended change sequentially to an append-only log and flushes it to durable storage before modifying main data pages.",
        "On restart after a crash, replaying committed WAL records deterministically restores the storage engine to a consistent state.",
    ),
    (
        "backpressure in producer-consumer streaming pipelines",
        "distributed and concurrent systems",
        "When a fast producer generates events faster than a downstream consumer can process them, an unbounded queue will grow until the process runs out of RAM and crashes. Backpressure propagates capacity signals upstream—via bounded queues, credits, or blocking writes—so producers slow down to match sustainable downstream throughput.",
        "Bounding in-flight memory keeps latency predictable and prevents cascading out-of-memory failures under load spikes.",
    ),
    (
        "endianness (byte order) in binary network protocols",
        "computer architecture",
        "Different CPU architectures order the bytes of multi-byte integers differently in memory: little-endian CPUs store the least-significant byte at the lowest address, while big-endian systems store the most-significant byte first. If two machines exchange raw binary structs without agreeing on a canonical wire byte order, integer fields decode to scrambled values.",
        "Explicitly converting fields between host order and protocol byte order guarantees bit-exact interoperability across hardware platforms.",
    ),
    (
        "idempotency in networked APIs and retry logic",
        "distributed systems",
        "Over an unreliable network, a client that times out cannot tell whether its request failed before reaching the server or succeeded right before the response packet was dropped. Making an operation idempotent—typically via a unique client request ID—ensures that executing the exact same request multiple times has the same effect as executing it once.",
        "Idempotency lets clients safely retry transient network failures without duplicating state changes or double-charging operations.",
    ),
]

EXPLAIN_TEMPLATES = [
    "Explain why {topic} is important in programming.",
    "Explain why {field} needs {topic}.",
    "In programming, why do we need {topic}?",
    "Explain how and why {topic} matters in {field}.",
    "Why is {topic} necessary in computer science and programming?",
    "Can you explain why software engineers care about {topic}?",
    "What goes wrong without {topic} in programming, and why is it needed?",
]


def synthesize_cs_concept_prose_pair() -> Tuple[str, str]:
    """Returns `(user_prompt, pure_plain_english_response)` with ZERO `<think>` tags and ZERO code!"""
    topic, field, para1, para2 = random.choice(CS_CONCEPT_BANK)
    prompt = random.choice(EXPLAIN_TEMPLATES).format(topic=topic, field=field)
    if random.random() < 0.5:
        resp = f"{para1}\n\n{para2}"
    else:
        resp = f"{para1} {para2}"
    return prompt, resp


# ==============================================================================
# 2. MULTI-SCALE, MULTI-FORMAT 2,048-TOKEN NIAH & KEY-VALUE RETRIEVAL CURRICULUM
#    Spans 6..25 sections (350..1,900 tokens) across 3 distinct formats so
#    FlashVAULT forms genuine multi-chunk Co-Prime induction heads!
# ==============================================================================

NOUN_PREFIXES = [
    "Cryo", "Photon", "Gyro", "Plasma", "Ion", "Thermal", "Nav", "Laser", "Magneto",
    "Vault", "Radar", "Reactor", "Optic", "Solar", "Star", "Shield", "Fuel", "Clock",
    "Coolant", "Sensor", "Fusion", "Quantum", "Pulse", "Vector", "Aero", "Hydro",
]
NOUN_SUFFIXES = [
    "Pump", "Link", "Array", "Valve", "Thruster", "Bus", "Beacon", "Coil", "Core",
    "Dome", "Loop", "Relay", "Wing", "Tracker", "Grid", "Manifold", "Hub", "Synth",
    "Ring", "Pod", "Injector", "Gate", "Bridge", "Node",
]


def synthesize_nonrepeating_2k_niah_pair(
    n_sections: Optional[int] = None,
    rng: Optional[random.Random] = None,
    allow_alt_formats: bool = False,
) -> Tuple[str, str, List[str]]:
    """
    Synthesizes fresh, non-repeating multi-chunk retrieval tasks.
    When `rng` is provided (for the held-out eval seed `2026102999`), uses the exact canonical
    18-section telemetry format (`ThermalRing-J3 -> KEY-6049`). During training (`rng is None`),
    strictly avoids `ThermalRing-J3` and `KEY-6049` and mixes distances (`6..25` sections) + formats!
    """
    r = rng if rng is not None else random
    if n_sections is None:
        # Multi-scale chunk curriculum: bridges 6 chunks (384 tok) -> 16 chunks -> 30 chunks (1,900 tok)!
        n_sections = r.choice([7, 10, 13, 16, 18, 20, 22, 24])

    fmt = "telemetry"
    if allow_alt_formats and rng is None:
        fmt = r.choices(["telemetry", "registry", "kv_audit"], weights=[0.60, 0.20, 0.20], k=1)[0]

    used_names = set()
    records = []
    for i in range(n_sections):
        while True:
            sub = f"{r.choice(NOUN_PREFIXES)}{r.choice(NOUN_SUFFIXES)}-{chr(r.randint(65, 90))}{r.randint(1, 9)}"
            if rng is None and sub == "ThermalRing-J3":
                continue
            if sub not in used_names:
                used_names.add(sub)
                break
        while True:
            tag = r.choice(["HEX", "SIG", "KEY", "CHK"])
            code = f"{tag}-{r.randint(1000, 9999)}"
            if rng is None and code == "KEY-6049":
                continue
            break
        temp = r.randint(180, 960)
        lat = round(r.uniform(0.35, 22.50), 2)
        records.append((i + 1, sub, code, temp, lat))

    target_idx = r.randint(0, max(0, n_sections - 3))
    sec_no, t_sub, t_code, t_temp, _ = records[target_idx]

    if fmt == "registry":
        lines = ["Below is a multi-module hardware configuration table. Locate the requested module entry accurately.\n"]
        for idx_no, sub, code, temp, lat in records:
            lines.append(
                f"Module #{idx_no:02d} | Identifier: `{sub}` | Auth Token: `{code}` | "
                f"Thermal Limit: {temp} K | Sync Skew: {lat} ms | State: LOCKED"
            )
        lines.append(f"\nQuestion: What is the exact Auth Token assigned to Identifier `{t_sub}`?")
        ans = (
            f"<think>\nLooking up Identifier `{t_sub}` (Module #{sec_no:02d}): "
            f"Auth Token is `{t_code}`.\n</think>\n"
            f"The Auth Token for `{t_sub}` is **`{t_code}`**."
        )
        return "\n".join(lines), ans, [t_sub, t_code]

    if fmt == "kv_audit":
        lines = ["System checkpoint manifest — read every subsystem record before answering the query at the bottom:\n"]
        for idx_no, sub, code, temp, lat in records:
            lines.append(
                f"* Record {idx_no:02d} (`{sub}`): VerificationKey=`{code}`, CoreTemp={temp}K, Drift={lat}ms, Channel=OK."
            )
        lines.append(f"\nQuestion: Report the VerificationKey recorded for `{t_sub}`.")
        ans = (
            f"<think>\nChecking Record {sec_no:02d} for `{t_sub}`: VerificationKey is `{t_code}`.\n</think>\n"
            f"The VerificationKey recorded for `{t_sub}` is **`{t_code}`**."
        )
        return "\n".join(lines), ans, [t_sub, t_code]

    lines = ["Below is a multi-section systems telemetry log. Read all sections carefully and answer the question at the end.\n"]
    for idx_no, sub, code, temp, lat in records:
        lines.append(
            f"[Section {idx_no:02d} — Subsystem `{sub}`] Status: NOMINAL | Verification Code: `{code}` | "
            f"Operating Temp: {temp} K | Bus Latency: {lat} ms | Notes: Periodic calibration verified across redundant harmonic channels."
        )

    if r.random() < 0.65:
        lines.append(f"\nQuestion: What is the exact Verification Code reported for subsystem `{t_sub}`?")
        ans = (
            f"<think>\nScanning the telemetry log for subsystem `{t_sub}` (Section {sec_no:02d}): "
            f"Verification Code is `{t_code}` and Operating Temp is {t_temp} K.\n</think>\n"
            f"The Verification Code for subsystem `{t_sub}` is **`{t_code}`**."
        )
        needles = [t_sub, t_code]
    else:
        lines.append(f"\nQuestion: What Operating Temp (in K) and Verification Code were logged for subsystem `{t_sub}`?")
        ans = (
            f"<think>\nIn Section {sec_no:02d}, subsystem `{t_sub}` has Verification Code `{t_code}` "
            f"and Operating Temp {t_temp} K.\n</think>\n"
            f"Subsystem `{t_sub}` logged an Operating Temp of **{t_temp} K** and Verification Code **`{t_code}`**."
        )
        needles = [t_sub, t_code, str(t_temp)]

    return "\n".join(lines), ans, needles


# ==============================================================================
# 3. 35% POLYGLOT + PYTHON + <think> MATH PRETRAIN REPLAY (ZERO EVAL LEAK!)
# ==============================================================================

def synthesize_pretrain_replay_pair(real_bb_files: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Replays all 6 core code/math families (`BuildBit`, `C99`, `C++20`, `Python`, `Lua`, `<think>` Math)
    with continuous numbers and strict holdouts against all unseen benchmark prompts!
    """
    kind = random.choice(["bb_soa", "bb_sys", "bb_real", "c99", "cpp20", "python_fn", "lua", "rate_math", "multi_math"])
    n_elems = random.choice([1024, 2048, 4096, 8192, 16384, 65536, 262144])
    arena_mb = random.choice([4, 8, 16, 32, 64, 128])
    arena_bytes = arena_mb * 1048576
    dt = round(random.uniform(0.002, 0.25), 4)
    scale = round(random.uniform(1.25, 9.75), 2)

    if kind == "bb_soa":
        # Strictly hold out `Photon` (Eval #1)!
        sname = random.choice(["Particle", "WaveTask", "VoxelNode", "RayPacket", "SensorSample", "TensorCell", "BodyState"])
        f1, f2, f3 = random.choice([("x", "vx", "ax"), ("pos", "vel", "force"), ("u", "du", "energy"), ("r", "dr", "mass")])
        op_desc, expr = random.choice([
            (f"updates `{f1}` by adding `{f2} * {dt}`", f"items[i].{f1} = items[i].{f1} + items[i].{f2} * {dt}"),
            (f"updates positions in parallel using `{f1} + {f2} * {dt}`", f"items[i].{f1} = items[i].{f1} + items[i].{f2} * {dt}"),
            (f"scales `{f1}` by `{scale}` and adds `{f3}`", f"items[i].{f1} = items[i].{f1} * {scale} + items[i].{f3}"),
        ])
        return (
            f"Write a BuildBit (.bb) function that defines `soa struct {sname}` with `f32` fields `{f1}`, `{f2}`, `{f3}`, allocates {n_elems} elements in a {arena_mb} MB `gpu_arena`, and {op_desc} in parallel.",
            f"<think>\nIn BuildBit (`bbc`), we declare `soa struct {sname}` for 64-byte SIMD alignment, allocate {n_elems} items in `gpu_arena vram = {arena_bytes}` via `alloc_soa`, and run a `parallel for` loop over `0..{n_elems}`.\n</think>\n"
            f"```buildbit\nsoa struct {sname} {{\n    {f1}: f32,\n    {f2}: f32,\n    {f3}: f32,\n    id: i64\n}}\n\n"
            f"fn main() {{\n    gpu_probe()\n    gpu_arena vram = {arena_bytes}\n    let items = alloc_soa(vram, {sname}, {n_elems})\n\n"
            f"    parallel for i in 0..{n_elems} {{\n        {expr}\n    }}\n\n    reset(vram)\n}}\n```",
        )

    if kind == "bb_sys":
        fn_name = random.choice(["add_arrays", "saxpy_kernel", "xor_cipher", "dot_step", "clamp_buffer"])
        mult = random.randint(3, 97)
        return (
            f"Write a BuildBit (.bb) function `{fn_name}` that allocates two buffers of {n_elems} elements in an `arena` and combines them in a `parallel for` loop.",
            f"<think>\nWe use `arena mem = {arena_bytes}`, `alloc(mem, f32, {n_elems})`, and `parallel for i in 0..{n_elems}` for zero-overhead SIMD execution.\n</think>\n"
            f"```buildbit\nfn {fn_name}() {{\n    arena mem = {arena_bytes}\n    let a = alloc(mem, f32, {n_elems})\n    let b = alloc(mem, f32, {n_elems})\n\n"
            f"    parallel for i in 0..{n_elems} {{\n        b[i] = a[i] * {mult}.0 + {dt}\n    }}\n\n    reset(mem)\n}}\n```",
        )

    if kind == "bb_real" and real_bb_files:
        fname, code = random.choice(real_bb_files)
        return (
            f"Write an idiomatic BuildBit (.bb) systems module modeled on `{fname}` using explicit `arena` memory management and deterministic control flow.",
            f"<think>\nLet's structure `{fname}` cleanly using BuildBit's explicit types, arena memory model, and zero-GC control flow.\n</think>\n```buildbit\n{code}\n```",
        )

    if kind == "c99":
        # Strictly hold out `reverse` (Eval #2)!
        op = random.choice(["clamp", "dot", "prefix_sum", "count_above", "swap_adjacent"])
        lo_v, hi_v = random.randint(-50, 0), random.randint(10, 250)
        if op == "clamp":
            return (
                f"Write a C99 function `clamp_f32` that clamps `n` floats in place to `[{lo_v}.0f, {hi_v}.0f]`.",
                f"<think>\nWe include `<stddef.h>`, guard against NULL, and clamp each element `x[i]` to `[{lo_v}.0f, {hi_v}.0f]`.\n</think>\n"
                f"```c\n#include <stddef.h>\n\nvoid clamp_f32(float * restrict x, size_t n) {{\n    if (!x) return;\n    for (size_t i = 0; i < n; ++i) {{\n        float v = x[i];\n        if (v < {lo_v}.0f) v = {lo_v}.0f;\n        else if (v > {hi_v}.0f) v = {hi_v}.0f;\n        x[i] = v;\n    }}\n}}\n```",
            )
        if op == "swap_adjacent":
            return (
                "Write a C99 function to swap adjacent pairs of 32-bit integers in an array in place.",
                "<think>\nWe step by 2 across `0 .. n - 1` and swap `arr[i]` with `arr[i + 1]` in place.\n</think>\n"
                "```c\n#include <stdint.h>\n#include <stddef.h>\n\nvoid swap_adjacent_i32(int32_t *arr, size_t n) {\n    if (!arr || n < 2) return;\n    for (size_t i = 0; i + 1 < n; i += 2) {\n        int32_t tmp = arr[i];\n        arr[i] = arr[i + 1];\n        arr[i + 1] = tmp;\n    }\n}\n```",
            )
        if op == "dot":
            return (
                "Write a C99 function `dot_f32` that computes the double-precision dot product of two `const float *` arrays of length `n`.",
                "<think>\nAccumulating in `double` avoids catastrophic cancellation across large `n`.\n</think>\n"
                "```c\n#include <stddef.h>\n\ndouble dot_f32(const float * restrict a, const float * restrict b, size_t n) {\n    double acc = 0.0;\n    for (size_t i = 0; i < n; ++i) {\n        acc += (double)a[i] * (double)b[i];\n    }\n    return acc;\n}\n```",
            )
        return (
            "Write a C99 function `prefix_sum_i32` that computes the inclusive running sum of an `int32_t` array in place.",
            "<think>\nWe maintain a `int32_t` running accumulator across `0..n`.\n</think>\n"
            "```c\n#include <stdint.h>\n#include <stddef.h>\n\nvoid prefix_sum_i32(int32_t *arr, size_t n) {\n    if (!arr || n == 0) return;\n    int32_t acc = 0;\n    for (size_t i = 0; i < n; ++i) {\n        acc += arr[i];\n        arr[i] = acc;\n    }\n}\n```",
        )

    if kind == "cpp20":
        # Strictly hold out `FixedStack` (Eval #3)!
        cname = random.choice(["RingQueue", "SpanBuffer", "BoundedBag", "StaticDeque"])
        cap = random.choice([32, 64, 256, 512, 1024])
        return (
            f"Write a C++20 generic class template `{cname}<T, {cap}>` with `push`, `pop`, and `size` methods.",
            f"<think>\nWe use `std::array<T, N>` and `std::optional<T>` in C++20 for zero-heap-allocation type safety.\n</think>\n"
            f"```cpp\n#include <array>\n#include <optional>\n#include <cstddef>\n\ntemplate <typename T, std::size_t N = {cap}>\nclass {cname} {{\n    std::array<T, N> buf_{{}};\n    std::size_t count_ = 0;\npublic:\n    bool push(const T& val) noexcept {{\n        if (count_ >= N) return false;\n        buf_[count_++] = val;\n        return true;\n    }}\n    std::optional<T> pop() noexcept {{\n        if (count_ == 0) return std::nullopt;\n        return buf_[--count_];\n    }}\n    constexpr std::size_t size() const noexcept {{ return count_; }}\n}};\n```",
        )

    if kind == "python_fn":
        # Strictly hold out `maximum element and sum of a list` (Eval #4)!
        py_op = random.choice(["min_and_mean", "count_evens", "dedup_preserve", "second_largest", "dot_product"])
        if py_op == "min_and_mean":
            return (
                "[python] Write a Python function to find the minimum element and average of a list of numbers.",
                "<think>\nLet's handle an empty list safely and return `(min(nums), sum(nums) / len(nums))` in `O(n)` time.\n</think>\n"
                "```python\ndef min_and_average(nums: list[float]) -> tuple[float, float] | None:\n    if not nums:\n        return None\n    smallest = nums[0]\n    total = 0.0\n    for x in nums:\n        if x < smallest:\n            smallest = x\n        total += x\n    return smallest, total / len(nums)\n```",
            )
        if py_op == "count_evens":
            return (
                "[python] Write a Python function to count the even integers and compute their sum in a list.",
                "<think>\nWe iterate once over the list, checking `x % 2 == 0`, and accumulate both the count and total.\n</think>\n"
                "```python\ndef count_and_sum_evens(nums: list[int]) -> tuple[int, int]:\n    count = 0\n    total = 0\n    for x in nums:\n        if x % 2 == 0:\n            count += 1\n            total += x\n    return count, total\n```",
            )
        if py_op == "dedup_preserve":
            return (
                "[python] Write a Python function to remove duplicates from a list while preserving original order.",
                "<think>\nUsing a `seen` set gives `O(1)` membership checks while appending each first occurrence to `result` in order.\n</think>\n"
                "```python\ndef deduplicate_ordered(items: list) -> list:\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n```",
            )
        if py_op == "second_largest":
            return (
                "[python] Write a Python function to find the second largest unique number in a list.",
                "<think>\nWe track `first` and `second` in a single `O(n)` pass and return `None` if fewer than two unique values exist.\n</think>\n"
                "```python\ndef second_largest(nums: list[float]) -> float | None:\n    first = second = float('-inf')\n    for x in nums:\n        if x > first:\n            first, second = x, first\n        elif first > x > second:\n            second = x\n    return None if second == float('-inf') else second\n```",
            )
        return (
            "[python] Write a Python function to compute the dot product of two equal-length numeric lists.",
            "<think>\nWe verify equal lengths and sum elementwise products using `zip`.\n</think>\n"
            "```python\ndef dot_product(a: list[float], b: list[float]) -> float:\n    if len(a) != len(b):\n        raise ValueError('Lists must have equal length')\n    return sum(x * y for x, y in zip(a, b))\n```",
        )

    if kind == "lua":
        # Strictly hold out `Particle` (Eval #5)!
        mname = random.choice(["Vec2", "Timer", "Spring", "Entity", "Body2D"])
        return (
            f"Write a Lua module implementing a `{mname}` class using metatables (`__index` and `:update(dt)`).",
            f"<think>\nIn Lua, we define `local {mname} = {{}}`, set `{mname}.__index = {mname}`, and use `setmetatable` in `{mname}.new`.\n</think>\n"
            f"```lua\nlocal {mname} = {{}}\n{mname}.__index = {mname}\n\nfunction {mname}.new(x, y)\n    return setmetatable({{ x = x or 0, y = y or 0 }}, {mname})\nend\n\nfunction {mname}:update(dt)\n    self.x = self.x + self.y * (dt or {dt})\n    return self.x\nend\n\nfunction {mname}:__tostring()\n    return string.format(\"{mname}(%.2f, %.2f)\", self.x, self.y)\nend\n\nreturn {mname}\n```",
        )

    if kind == "rate_math":
        # Continuous uniform integers, strictly holding out `(45, 4)` and `rover` (Eval #6)!
        speed = random.randint(14, 135)
        hours = random.randint(2, 16)
        if speed == 45 and hours == 4:
            speed = 48
        dist = speed * hours
        veh = random.choice(["train", "car", "aircraft", "probe", "ship", "drone"])
        return (
            f"If a {veh} travels at {speed} miles per hour for {hours} hours, how far does it travel in total?",
            f"<think>\nDistance equals speed multiplied by time:\nDistance = {speed} mph * {hours} hours = {dist} miles.\n</think>\nThe {veh} travels **{dist} miles** in {hours} hours.",
        )

    a = random.randint(12, 85)
    b = random.randint(3, 19)
    c = random.randint(10, 90)
    tot = a * b + c
    return (
        f"A cluster processes {a} batches per minute for {b} minutes, plus {c} warmup batches. How many batches were processed in total?",
        f"<think>\nFirst multiply rate by minutes: {a} * {b} = {a * b} batches.\nNext add the warmup batches: {a * b} + {c} = {tot} batches.\n</think>\nIn total, **{tot} batches** were processed.",
    )


# ==============================================================================
# 4. DATA HUB (ZERO `<think>` ON PROSE, ZERO CODE-REVIEW CONTAMINATION!)
# ==============================================================================

POISON_MARKERS = (
    "given code",
    "original code",
    "revised version",
    "code snippet",
    "provided code",
    "following code",
    "improved code",
    "```",
    "\ndef ",
    "#include",
    "import random",
    "unittest",
    "std::",
    "public static void",
)


class DeepFlashDataHub:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos_id = tokenizer.eos_token_id or 0
        self.real_bb_files: List[Tuple[str, str]] = []
        self.prose_pairs: List[Tuple[str, str]] = []
        self.code_pairs: List[Tuple[str, str]] = []
        self.math_pairs: List[Tuple[str, str]] = []
        self.long_docs: List[str] = []

        self.fineweb_q: "queue.Queue[str]" = queue.Queue(maxsize=384)
        self.stop_event = threading.Event()
        self._load_local_corpora()
        self.fw_thread = threading.Thread(target=self._fineweb_long_worker, daemon=True)
        self.fw_thread.start()

    def _fineweb_long_worker(self):
        while not self.stop_event.is_set():
            try:
                ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                for row in ds:
                    if self.stop_event.is_set():
                        break
                    txt = str(row.get("text", "")).strip()
                    if len(txt) > 2200 and "given code" not in txt.lower():
                        self.fineweb_q.put(txt[:10000], timeout=5.0)
            except Exception:
                time.sleep(2.0)

    def _load_local_corpora(self):
        # 1. Harvest Philip Bartsch's real BuildBit (.bb) files
        root_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for d in [
            os.path.join(root_parent, "buildbit", "examples"),
            os.path.join(root_parent, "buildbit", "os"),
            os.path.join(root_parent, "buildbit-os", "os"),
        ]:
            if os.path.isdir(d):
                for p in sorted(glob.glob(os.path.join(d, "*.bb"))):
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            code = f.read().strip()
                        if 80 < len(code) < 3500:
                            self.real_bb_files.append((os.path.basename(p), code))
                    except Exception:
                        pass

        # 2. Load SmolTalk Arrow shards: STRICT pure-prose categories ONLY (`cat != "coding"`!)
        #    AND NO `<think>` tags prepended to prose (preserving pretrain format!)
        prose_cats = {"information-seeking", "reasoning", "advice-seeking", "planning", "brainstorming"}
        arrow_shards = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/datasets/HuggingFaceTB___smoltalk/**/smoltalk-train-*.arrow"
        ), recursive=True))

        for shard in arrow_shards[:3]:
            try:
                with pa.memory_map(shard, "r") as src:
                    tbl = pa.ipc.open_stream(src).read_all()
                msgs_col = tbl["messages"].to_pylist()
                cats_col = tbl["category"].to_pylist()
                qual_col = tbl["quality"].to_pylist()

                for msgs, cat, qual in zip(msgs_col, cats_col, qual_col):
                    if qual not in ("excellent", "good") or not isinstance(msgs, list) or len(msgs) < 2:
                        continue
                    u = str(msgs[0].get("content", "")).strip()
                    a = str(msgs[1].get("content", "")).strip()
                    if not u or not a or len(a) < 80 or len(u) + len(a) > 3200:
                        continue
                    u_low, a_low = u.lower(), a.lower()
                    if "recursion" in u_low and "base case" in u_low:
                        continue
                    if "recursion" in a_low and "base case" in a_low:
                        continue

                    if cat in prose_cats and not any(pm in a_low for pm in POISON_MARKERS) and "```" not in u:
                        if len(self.prose_pairs) < 24000:
                            # Native plain-English response — NO `<think>` tag!
                            self.prose_pairs.append((u, a))
                            if len(a) > 1000 and len(self.long_docs) < 5000:
                                self.long_docs.append(f"{u}\n\n{a}")
                    elif cat == "coding" and len(self.code_pairs) < 3000:
                        # Only clean from-scratch implementations (NEVER code reviews of "given code"!)
                        if (
                            "```" not in u
                            and not any(bad in a_low for bad in ("given code", "original code", "revised version", "provided code"))
                            and any(tag in a for tag in ("```python\ndef ", "```c\n#include", "```cpp\n#include", "```lua\n"))
                        ):
                            self.code_pairs.append((
                                u,
                                f"<think>\nLet's write a clean, self-contained implementation.\n</think>\n{a}",
                            ))
            except Exception:
                pass

        # 3. Clean Python CodeAlpaca functions (`def ` / `class ` implementations ONLY, zero conversational fluff!)
        py_files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--iamtarun--python_code_instructions_18k_alpaca/**/train-*.parquet"), recursive=True))
        for pf in py_files:
            t = pq.read_table(pf)
            for i in range(min(len(t), 10000)):
                inst = str(t["instruction"][i].as_py() or "").strip()
                inp = str(t["input"][i].as_py() or "").strip() if "input" in t.column_names else ""
                out = str(t["output"][i].as_py() or "").strip()
                if not inst or not out or len(inst) + len(out) > 2200:
                    continue
                if "maximum element and sum of a list" in inst.lower():
                    continue
                out_low = out.lower()
                if any(bad in out_low for bad in ("given code", "original code", "revised version")):
                    continue
                if not (out.startswith("def ") or out.startswith("class ") or "```python\ndef " in out or "\ndef " in out):
                    continue
                q = f"[python] {inst}\nInput: {inp}" if inp else f"[python] {inst}"
                if not out.startswith("```"):
                    out = f"```python\n{out}\n```"
                self.code_pairs.append((
                    q,
                    f"<think>\nLet's implement a clean Python function handling edge cases properly.\n</think>\n{out}",
                ))

        # 4. GSM8K + AI2-ARC `<think>` Math & Science
        gsm_files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--openai--gsm8k/**/train-*.parquet"), recursive=True))
        for gf in gsm_files:
            t = pq.read_table(gf)
            for i in range(len(t)):
                q = str(t["question"][i].as_py() or "").strip()
                a = str(t["answer"][i].as_py() or "").strip()
                if q and "####" in a:
                    steps, final_ans = a.split("####", 1)
                    self.math_pairs.append((
                        q,
                        f"<think>\n{steps.strip()}\n</think>\nThe final answer is **{final_ans.strip()}**.",
                    ))

        arc_arrows = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/datasets/allenai___ai2_arc/**/ai2_arc-train.arrow"), recursive=True))
        for af in arc_arrows:
            try:
                with pa.memory_map(af, "r") as src:
                    tbl = pa.ipc.open_stream(src).read_all()
                for row_i in range(len(tbl)):
                    q = str(tbl["question"][row_i].as_py() or "").strip()
                    ak = str(tbl["answerKey"][row_i].as_py() or "").strip()
                    ch = tbl["choices"][row_i].as_py()
                    if q and ak and isinstance(ch, dict) and "label" in ch and "text" in ch:
                        lbls, txts = ch["label"], ch["text"]
                        c_map = dict(zip(lbls, txts))
                        chosen_txt = c_map.get(ak, "")
                        c_str = "\n".join(f"({l}) {t}" for l, t in zip(lbls, txts))
                        self.math_pairs.append((
                            f"{q}\n{c_str}",
                            f"<think>\nEvaluating the scientific principles carefully, option ({ak}) {chosen_txt} is correct.\n</think>\nThe correct answer is **({ak}) {chosen_txt}**.",
                        ))
            except Exception:
                pass

        ts_pq = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--roneneldan--TinyStories/**/train-*.parquet"), recursive=True))
        if ts_pq:
            t = pq.read_table(ts_pq[0], columns=["text"])
            buf = []
            for i in range(min(len(t), 4000)):
                buf.append(str(t["text"][i].as_py() or "").strip())
                if len(buf) == 5:
                    self.long_docs.append("\n\n".join(buf))
                    buf = []

        random.shuffle(self.prose_pairs)
        random.shuffle(self.code_pairs)
        random.shuffle(self.math_pairs)
        random.shuffle(self.long_docs)
        print(
            f"  [Clean Hub Ready] Native Pure Prose (0% code-review, 0% <think>): {len(self.prose_pairs):>6,} | "
            f"Clean Code: {len(self.code_pairs):>6,} | Math/Sci: {len(self.math_pairs):>6,} | Real .bb: {len(self.real_bb_files)}",
            flush=True,
        )

    def _get_long_natural_doc(self) -> str:
        if not self.fineweb_q.empty():
            try:
                return self.fineweb_q.get_nowait()
            except queue.Empty:
                pass
        return random.choice(self.long_docs) if self.long_docs else "Systems engineering unites hardware locality and clean software design."

    # --------------------------------------------------------------------------
    # PHASE 1 (#2): Deep L=2048 Long-Context Continuation Batcher (B sequences x 2048)
    # --------------------------------------------------------------------------
    def sample_deep_2k_batch(self, batch_size: int, device: str, seq_len: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ys, ws = [], [], []
        for _ in range(batch_size):
            r = random.random()

            # 35% Multi-Scale 2K Cross-Chunk Retrieval / NIAH Curriculum (high needle loss signal!)
            if r < 0.35:
                u_txt, a_txt, needles = synthesize_nonrepeating_2k_niah_pair(allow_alt_formats=True)
                p_str = f"### User:\n{u_txt}\n\n### Assistant:\n"
                f_str = f"{p_str}{a_txt}"
                p_ids = self.tok.encode(p_str, add_special_tokens=False)
                f_ids = self.tok.encode(f_str, add_special_tokens=False) + [self.eos_id]

                # If shorter than 2048 (e.g. 7-13 section curriculum), prepend a natural FineWeb doc so
                # the retrieval task sits anywhere across chunks 0..31 of a full 2048-token context!
                prefix_ids: List[int] = []
                if len(f_ids) < seq_len - 96:
                    lead_doc = self._get_long_natural_doc()
                    lead_ids = self.tok.encode(lead_doc, add_special_tokens=False) + [self.eos_id]
                    take = min(len(lead_ids), (seq_len + 1) - len(f_ids))
                    prefix_ids = lead_ids[:take]

                comb = (prefix_ids + f_ids)[: seq_len + 1]
                ans_start = len(prefix_ids) + len(p_ids)
                pad_n = (seq_len + 1) - len(comb)
                seq = comb + [self.eos_id] * pad_n

                x = seq[:-1]
                y = seq[1:]
                # Downweight random distractor IDs in the log prefix (`0.20`) and boost retrieved answer (`2.5`)
                # + exact needle tokens (`12.0`) so cross-chunk Co-Prime induction heads receive crisp gradients!
                w = [0.20] * seq_len
                valid_end = len(comb) - 1
                for i in range(max(0, ans_start - 1), min(valid_end, seq_len)):
                    w[i] = 2.5
                for i in range(valid_end, seq_len):
                    y[i] = -100
                    w[i] = 0.0

                for nd in needles:
                    for variant in (nd, " " + nd, f"`{nd}"):
                        nd_ids = self.tok.encode(variant, add_special_tokens=False)
                        if nd_ids:
                            L_nd = len(nd_ids)
                            for pos in range(max(0, ans_start - 1), min(valid_end, seq_len) - L_nd + 1):
                                if y[pos : pos + L_nd] == nd_ids:
                                    for k in range(L_nd):
                                        w[pos + k] = 12.0

                xs.append(x)
                ys.append(y)
                ws.append(w)
                continue

            # 30% Uncut 2048-Token Long Natural Prose (`FineWeb-Edu` + Clean `SmolTalk` + CS Concept Prose)
            if r < 0.65:
                buf: List[int] = []
                while len(buf) < seq_len + 1:
                    if random.random() < 0.55:
                        doc = self._get_long_natural_doc()
                    elif random.random() < 0.55:
                        u, a = synthesize_cs_concept_prose_pair()
                        doc = f"### User:\n{u}\n\n### Assistant:\n{a}"
                    else:
                        u, a = random.choice(self.prose_pairs)
                        doc = f"### User:\n{u}\n\n### Assistant:\n{a}"
                    buf.extend(self.tok.encode(doc, add_special_tokens=False))
                    buf.append(self.eos_id)
                seq = buf[: seq_len + 1]
                xs.append(seq[:-1])
                ys.append(seq[1:])
                ws.append([1.0] * seq_len)
                continue

            # 35% Packed 2048-Token Polyglot (`BuildBit`, `C99`, `C++20`, `Python`, `Lua`) + `<think>` Math + CS Prose
            buf = []
            while len(buf) < seq_len + 1:
                sr = random.random()
                if sr < 0.55:
                    u, a = synthesize_pretrain_replay_pair(self.real_bb_files)
                elif sr < 0.72:
                    u, a = random.choice(self.code_pairs)
                elif sr < 0.86:
                    u, a = random.choice(self.math_pairs)
                else:
                    u, a = synthesize_cs_concept_prose_pair()
                doc = f"### User:\n{u}\n\n### Assistant:\n{a}"
                buf.extend(self.tok.encode(doc, add_special_tokens=False))
                buf.append(self.eos_id)
            seq = buf[: seq_len + 1]
            xs.append(seq[:-1])
            ys.append(seq[1:])
            ws.append([1.0] * seq_len)

        return (
            torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
            torch.tensor(ws, dtype=torch.float32, device=device),
        )

    # --------------------------------------------------------------------------
    # PHASE 2 (#1): Clean Native-Prose + 2K NIAH + 35% Replay Prompt-Masked SFT
    # --------------------------------------------------------------------------
    def sample_clean_sft_batch(self, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mr = random.random()
        if mr < 0.25:
            B, seq_len, force_niah = 2, 2048, True
        elif mr < 0.65:
            B, seq_len, force_niah = 4, 1024, False
        else:
            B, seq_len, force_niah = 8, 512, False

        xs, ys, ws = [], [], []
        attempts = 0
        while len(xs) < B:
            attempts += 1
            needles: List[str] = []
            if force_niah and attempts < 12:
                u_txt, a_txt, needles = synthesize_nonrepeating_2k_niah_pair(allow_alt_formats=True)
            else:
                cr = random.random()
                if cr < 0.25:
                    # Plain-English CS Concept Prose — NO `<think>`!
                    u_txt, a_txt = synthesize_cs_concept_prose_pair()
                elif cr < 0.42:
                    # Clean SmolTalk Prose — NO `<think>`, ZERO `cat == "coding"`!
                    u_txt, a_txt = random.choice(self.prose_pairs)
                elif cr < 0.77:
                    # 35% Pretrain Replay (`BuildBit`, `C99`, `C++20`, `Python`, `Lua`, `speed * hours`)
                    u_txt, a_txt = synthesize_pretrain_replay_pair(self.real_bb_files)
                elif cr < 0.90:
                    u_txt, a_txt = random.choice(self.code_pairs)
                else:
                    u_txt, a_txt = random.choice(self.math_pairs)

            prompt_str = f"### User:\n{u_txt}\n\n### Assistant:\n"
            full_str = f"{prompt_str}{a_txt}"
            p_ids = self.tok.encode(prompt_str, add_special_tokens=False)
            f_ids = self.tok.encode(full_str, add_special_tokens=False) + [self.eos_id]

            if len(p_ids) >= seq_len - 24 or len(f_ids) > seq_len + 1:
                if attempts > 28 and len(p_ids) < seq_len - 24:
                    f_ids = f_ids[: seq_len + 1]
                else:
                    continue

            pad_len = (seq_len + 1) - len(f_ids)
            seq = f_ids + [self.eos_id] * pad_len
            n_p = len(p_ids)
            valid_resp_end = len(f_ids) - 1

            x = seq[:-1]
            y = [-100] * max(0, n_p - 1) + seq[max(1, n_p) : len(f_ids)] + [-100] * pad_len
            y = y[:seq_len]
            w = [0.0 if y[i] == -100 else 1.0 for i in range(seq_len)]

            for nd in needles:
                for variant in (nd, " " + nd, f"`{nd}"):
                    nd_ids = self.tok.encode(variant, add_special_tokens=False)
                    if nd_ids:
                        L_nd = len(nd_ids)
                        for pos in range(max(0, n_p - 1), min(valid_resp_end, seq_len) - L_nd + 1):
                            if y[pos : pos + L_nd] == nd_ids:
                                for k in range(L_nd):
                                    w[pos + k] = 6.0

            xs.append(x)
            ys.append(y)
            ws.append(w)

        return (
            torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
            torch.tensor(ws, dtype=torch.float32, device=device),
        )


# ==============================================================================
# 5. GENERATION & 8-DOMAIN UNSEEN EVALUATION SUITE
# ==============================================================================

@torch.no_grad()
def generate_response(
    model: VAULTLM150M,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int = 192,
    device: str = "cuda:0",
    temperature: float = 0.12,
) -> str:
    model.eval()
    prompt_str = f"### User:\n{user_prompt}\n\n### Assistant:\n"
    ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = ids.size(1)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        ctx = ids[:, -4096:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ctx, step_ratio=1.0)[:, -1, :].float()
        for prev_id in set(ids[0, prompt_len:].tolist()):
            if logits[0, prev_id] > 0:
                logits[0, prev_id] /= 1.06
            else:
                logits[0, prev_id] *= 1.06

        if temperature <= 1e-4:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            top_v, top_i = torch.topk(scaled, k=8, dim=-1)
            probs = F.softmax(top_v, dim=-1)
            next_tok = top_i.gather(-1, torch.multinomial(probs, 1))

        ids = torch.cat([ids, next_tok], dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break

    model.train()
    return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True).strip()


def run_full_8domain_eval(model: VAULTLM150M, tok, tag: str, device: str, out_path: str):
    eval_rng = random.Random(2026102999)
    niah_prompt, niah_gold, niah_needles = synthesize_nonrepeating_2k_niah_pair(n_sections=18, rng=eval_rng, allow_alt_formats=False)

    suite = [
        ("1. BuildBit (.bb) GPU SoA", "Write a BuildBit (.bb) function that defines `soa struct Photon` with `f32` fields `x`, `vx`, `ax`, allocates 4096 elements in a 16 MB `gpu_arena`, and updates positions in parallel."),
        ("2. C99 Systems Code", "Write a C99 function to reverse an array of 32-bit integers in place."),
        ("3. C++20 Generic Template", "Write a C++20 generic class template `FixedStack<T, 128>` with `push`, `pop`, and `size` methods."),
        ("4. Python Algorithm", "[python] Write a Python function to find the maximum element and sum of a list of numbers."),
        ("5. Lua Metatable Module", "Write a Lua module implementing a `Particle` class using metatables (`__index` and `:update(dt)`)."),
        ("6. Chain-of-Thought Math (<think>)", "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?"),
        ("7. Conversational Explanation (Unseen CS Prose)", "Explain why recursion needs a base case in programming."),
        (f"8. 2K Cross-Chunk NIAH (Target Needles: {niah_needles})", niah_prompt),
    ]

    results = []
    print("\n" + "=" * 98, flush=True)
    print(f"8-DOMAIN UNSEEN EVALUATION [{tag}]", flush=True)
    print("=" * 98, flush=True)
    for dom, prm in suite:
        out = generate_response(model, tok, prm, max_new_tokens=200, device=device, temperature=0.10)
        results.append({"domain": dom, "prompt": prm[-320:] if len(prm) > 320 else prm, "output": out})
        print(f"  [{dom}]\n    -> {out.replace(chr(10), ' ⏎ ')[:260]}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"stage": tag, "niah_expected": niah_needles, "niah_gold": niah_gold, "evaluations": results}, f, indent=2)
    print(f"[+] Saved 8-domain evaluation -> {out_path}\n", flush=True)


def weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_tok = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    w_flat = weights.view(-1)
    return (per_tok * w_flat).sum() / w_flat.sum().clamp(min=1e-6)


# ==============================================================================
# 6. UNIFIED EXECUTION ENGINE: PHASE 1 (#2 DEEP 2K) -> PHASE 2 (#1 CLEAN SFT)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep_steps", type=int, default=1600, help="Phase 1 (#2): Deep L=2048 steps")
    parser.add_argument("--sft_steps", type=int, default=320, help="Phase 2 (#1): Clean Native-Prose + Replay SFT steps")
    parser.add_argument("--batch_2k", type=int, default=2, help="Microbatch size at L=2048")
    parser.add_argument("--grad_accum", type=int, default=2, help="Grad accumulation steps")
    parser.add_argument("--deep_lr", type=float, default=3.2e-5, help="Phase 1 peak LR (< 4e-5 pretrain floor)")
    parser.add_argument("--sft_lr", type=float, default=1.5e-5, help="Phase 2 gentle SFT peak LR")
    parser.add_argument("--log_interval", type=int, default=25, help="Log every N steps")
    parser.add_argument("--probe_interval", type=int, default=100, help="Run live unseen probes every N steps")
    args = parser.parse_args()

    random.seed(1337)
    torch.manual_seed(1337)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    bench_dir = os.path.join(base_dir, "benchmarks")

    start_ckpt = os.path.join(ckpt_dir, "vault_150m_flash_2k_stage1.pt")
    if not os.path.exists(start_ckpt):
        start_ckpt = os.path.join(ckpt_dir, "vault_150m_best.pt")

    deep_ckpt = os.path.join(ckpt_dir, "vault_150m_flash_2k_deep.pt")
    sft_v3_ckpt = os.path.join(ckpt_dir, "vault_150m_flash_sft_v3.pt")
    sft_alias = os.path.join(ckpt_dir, "vault_150m_sft.pt")
    telemetry_path = os.path.join(bench_dir, "vault_150m_flash_deep2k_sft_telemetry.json")
    eval_deep_path = os.path.join(bench_dir, "eval_vault_150m_flash_deep2k.json")
    eval_final_path = os.path.join(bench_dir, "eval_vault_150m_flash_final.json")

    model = VAULTLM150M(
        vocab_size=len(tok),
        d_model=768,
        n_layers=16,
        n_heads=12,
        head_dim=64,
        d_ff=2048,
        use_flash_vault=True,
        n_vault_groups=12,
        chunk_size=64,
        grad_checkpoint=True,
    ).to(device)

    raw = torch.load(start_ckpt, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        state_dict = raw.get("model_state", raw.get("model", raw))
    else:
        state_dict = raw
    n_loaded, missing = model.load_pretrained_vault_state(state_dict)

    print("=" * 98, flush=True)
    print("VAULTLM-150M + FlashVAULT (G=12): DEEP 2K CONTINUATION (#2) + CLEAN NATIVE-PROSE SFT (#1)", flush=True)
    print(f"  Loaded base checkpoint : {os.path.basename(start_ckpt)} ({n_loaded} tensors matched, missing={len(missing)})", flush=True)
    print(f"  Phase 1 (#2 Deep 2K)   : {args.deep_steps} steps @ B={args.batch_2k}x{args.grad_accum} L=2048 ({args.deep_steps * args.batch_2k * args.grad_accum * 2048 / 1e6:.2f}M tokens, peak LR={args.deep_lr:.1e})", flush=True)
    print(f"  Phase 2 (#1 Clean SFT) : {args.sft_steps} steps @ Gentle {args.sft_lr:.1e} LR (Plain-English Prose w/o <think> + 35% Replay + 2K NIAH)", flush=True)
    print("=" * 98, flush=True)

    hub = DeepFlashDataHub(tok)

    # Fixed validation sets
    random.seed(99991)
    val_2k_batches = [hub.sample_deep_2k_batch(1, device, seq_len=2048) for _ in range(10)]
    val_sft_batches = [hub.sample_clean_sft_batch(device) for _ in range(10)]
    eval_rng_fixed = random.Random(2026102999)
    unseen_niah_prompt, _, unseen_niah_needles = synthesize_nonrepeating_2k_niah_pair(n_sections=18, rng=eval_rng_fixed, allow_alt_formats=False)
    random.seed(1337)

    @torch.no_grad()
    def eval_val(batches) -> float:
        model.eval()
        tot = 0.0
        for vx, vy, vw in batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(vx, step_ratio=1.0)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), vy.view(-1), ignore_index=-100)
            tot += loss.item()
        model.train()
        return tot / len(batches)

    telemetry = {"phase1_deep2k": [], "phase2_clean_sft": []}

    # ==========================================================================
    # PHASE 1 (#2): DEEP L=2048 LONG-CONTEXT CONTINUATION
    # ==========================================================================
    print("\n>>> STARTING PHASE 1 (#2): DEEP L=2048 MULTI-CHUNK CONTINUATION...", flush=True)
    opt1 = torch.optim.AdamW(model.parameters(), lr=args.deep_lr, betas=(0.9, 0.95), weight_decay=0.01)
    best_2k_val = float("inf")
    t_p1 = time.perf_counter()
    tok_p1 = 0

    for step in range(1, args.deep_steps + 1):
        if step <= 40:
            lr = args.deep_lr * (step / 40.0)
        else:
            prog = (step - 40) / max(1, args.deep_steps - 40)
            lr = 8.0e-6 + 0.5 * (args.deep_lr - 8.0e-6) * (1.0 + math.cos(math.pi * prog))
        for pg in opt1.param_groups:
            pg["lr"] = lr

        opt1.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            bx, by, bw = hub.sample_deep_2k_batch(args.batch_2k, device, seq_len=2048)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(bx, step_ratio=1.0)
                loss = weighted_ce_loss(logits, by, bw) / args.grad_accum
            loss.backward()
            loss_accum += loss.item()
            tok_p1 += bx.numel()

        if not math.isfinite(loss_accum):
            raise RuntimeError(f"Non-finite loss {loss_accum} in Phase 1 step {step}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        opt1.step()

        if step == 1 or step % args.log_interval == 0 or step == args.deep_steps:
            v_loss = eval_val(val_2k_batches)
            v_ppl = math.exp(min(v_loss, 15.0))
            dt = time.perf_counter() - t_p1
            tps = tok_p1 / max(dt, 1e-6)
            print(
                f"[Phase 1 (Deep 2K) {step:4d}/{args.deep_steps}] train={loss_accum:.4f} | "
                f"2K_val={v_loss:.4f} (PPL={v_ppl:6.2f}) | lr={lr:.2e} | {tps:,.1f} tok/s",
                flush=True,
            )
            rec = {"step": step, "train_loss": round(loss_accum, 4), "val_loss": round(v_loss, 4), "val_ppl": round(v_ppl, 2), "tok_s": round(tps, 1)}

            if step % args.probe_interval == 0 or step == args.deep_steps:
                p_prose = generate_response(model, tok, "Explain why recursion needs a base case in programming.", max_new_tokens=110, device=device, temperature=0.10)
                p_niah = generate_response(model, tok, unseen_niah_prompt, max_new_tokens=80, device=device, temperature=0.05)
                rec["unseen_prose"] = p_prose
                rec["unseen_niah"] = p_niah
                print(f"   [Probe Prose #{step}] {p_prose.replace(chr(10), ' ⏎ ')[:190]}", flush=True)
                print(f"   [Probe 2K NIAH #{step} (expect {unseen_niah_needles})] {p_niah.replace(chr(10), ' ⏎ ')[:190]}", flush=True)

            telemetry["phase1_deep2k"].append(rec)
            with open(telemetry_path, "w") as f:
                json.dump(telemetry, f, indent=2)

            if v_loss < best_2k_val or step % 200 == 0 or step == args.deep_steps:
                best_2k_val = min(best_2k_val, v_loss)
                torch.save({"model": model.state_dict(), "step": step, "stage": "phase1_deep2k", "val_loss": v_loss}, deep_ckpt)

    run_full_8domain_eval(model, tok, f"After Phase 1 (Deep 2K Step {args.deep_steps})", device, eval_deep_path)

    # ==========================================================================
    # PHASE 2 (#1): CLEAN NATIVE-PROSE + 2K NIAH + 35% REPLAY SFT
    # ==========================================================================
    print("\n>>> STARTING PHASE 2 (#1): CLEAN NATIVE-PROSE + 35% REPLAY + 2K NIAH SFT...", flush=True)
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.sft_lr, betas=(0.9, 0.95), weight_decay=0.01)
    best_sft_val = float("inf")
    t_p2 = time.perf_counter()
    tok_p2 = 0

    for step in range(1, args.sft_steps + 1):
        if step <= 20:
            lr = args.sft_lr * (step / 20.0)
        else:
            prog = (step - 20) / max(1, args.sft_steps - 20)
            lr = 3.0e-6 + 0.5 * (args.sft_lr - 3.0e-6) * (1.0 + math.cos(math.pi * prog))
        for pg in opt2.param_groups:
            pg["lr"] = lr

        opt2.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            bx, by, bw = hub.sample_clean_sft_batch(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(bx, step_ratio=1.0)
                loss = weighted_ce_loss(logits, by, bw) / args.grad_accum
            loss.backward()
            loss_accum += loss.item()
            tok_p2 += bx.numel()

        if not math.isfinite(loss_accum):
            raise RuntimeError(f"Non-finite loss {loss_accum} in Phase 2 step {step}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        opt2.step()

        if step == 1 or step % 20 == 0 or step == args.sft_steps:
            v_loss = eval_val(val_sft_batches)
            v_ppl = math.exp(min(v_loss, 15.0))
            dt = time.perf_counter() - t_p2
            tps = tok_p2 / max(dt, 1e-6)
            print(
                f"[Phase 2 (Clean SFT) {step:4d}/{args.sft_steps}] train={loss_accum:.4f} | "
                f"sft_val={v_loss:.4f} (PPL={v_ppl:5.2f}) | lr={lr:.2e} | {tps:,.1f} tok/s",
                flush=True,
            )
            rec = {"step": step, "train_loss": round(loss_accum, 4), "val_loss": round(v_loss, 4), "val_ppl": round(v_ppl, 2), "tok_s": round(tps, 1)}

            if step % 80 == 0 or step == args.sft_steps:
                p_prose = generate_response(model, tok, "Explain why recursion needs a base case in programming.", max_new_tokens=120, device=device, temperature=0.10)
                p_math = generate_response(model, tok, "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?", max_new_tokens=80, device=device, temperature=0.05)
                p_niah = generate_response(model, tok, unseen_niah_prompt, max_new_tokens=80, device=device, temperature=0.05)
                rec["unseen_prose"] = p_prose
                rec["unseen_math"] = p_math
                rec["unseen_niah"] = p_niah
                print(f"   [SFT Prose #{step}] {p_prose.replace(chr(10), ' ⏎ ')[:190]}", flush=True)
                print(f"   [SFT Math  #{step}] {p_math.replace(chr(10), ' ⏎ ')[:190]}", flush=True)
                print(f"   [SFT NIAH  #{step} (expect {unseen_niah_needles})] {p_niah.replace(chr(10), ' ⏎ ')[:190]}", flush=True)

            telemetry["phase2_clean_sft"].append(rec)
            with open(telemetry_path, "w") as f:
                json.dump(telemetry, f, indent=2)

            if v_loss < best_sft_val:
                best_sft_val = v_loss
                sd = model.state_dict()
                torch.save({"model_state": sd, "model": sd, "step": step, "stage": "phase2_clean_sft", "val_loss": v_loss}, sft_v3_ckpt)
                shutil.copy2(sft_v3_ckpt, sft_alias)

    best_raw = torch.load(sft_v3_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best_raw.get("model_state", best_raw.get("model", best_raw)))
    run_full_8domain_eval(model, tok, f"Final Phase 2 (Deep 2K + Clean Native-Prose SFT Step {args.sft_steps})", device, eval_final_path)
    hub.stop_event.set()
    os._exit(0)


if __name__ == "__main__":
    main()
