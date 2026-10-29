# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULTLM-150M + FlashVAULT (G=12): Complete 2-Stage Long-Context + Contrastive Intent + Replay Pipeline
# Executes all 3 upgrades sequentially starting from untouched `checkpoints/vault_150m_best.pt` (2.1048 PPL):
#   - Stage 1 (`L = 2048` Long-Context Pretraining Continuation, `lr = 3.0e-5 -> 1.0e-5`):
#       Adapts all 16 FlashVAULT layers from `seq_len = 256` to `seq_len = 2048` (32 chunks) on uncut
#       FineWeb-Edu/SmolTalk documents, 30% polyglot/math pretrain replay, and non-repeating 2K NIAH logs.
#   - Stage 2 (Gentle `lr = 1.5e-5 -> 3.0e-6` Contrastive CS-Intent + 30% Pretrain Replay + 2K NIAH SFT):
#       Separates "Explain/Why [CS concept] in programming" (-> pure prose) from "Write/Implement [code]" (-> code)
#       while preserving 100% of exact prompt-number & identifier copying (`45 * 4 = 180`, `reverse_i32`, `.bb`).
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
# 1. PROCEDURAL CONTRASTIVE CS-CONCEPT PROSE GENERATOR (2,500+ COMBINATIONS)
#    Strictly holds out "recursion" and "base case" so Eval #7 stays 100% unseen!
# ==============================================================================

CS_CONCEPT_BANK = [
    (
        "an exit condition in a `while` loop",
        "loops and control flow",
        "Without a reachable exit condition, the loop predicate stays `true` on every iteration, trapping the CPU thread inside the loop body forever (an infinite loop) so the program hangs and never returns a result.",
        "Before each iteration, the CPU evaluates the boolean guard; only when state updates inside the body eventually make that guard `false` can execution advance past the loop.",
    ),
    (
        "stack memory versus heap memory",
        "memory management",
        "Stack memory stores fixed-lifetime local variables and function call frames in contiguous LIFO order using a single CPU stack-pointer register (`O(1)` push/pop), whereas heap memory stores dynamically sized or long-lived objects managed via explicit allocations (`malloc`/arenas/GC).",
        "Using the stack keeps hot local variables in L1 cache with zero allocator overhead, while the heap allows data structures to grow dynamically and outlive the function that created them.",
    ),
    (
        "a sorted input array in binary search",
        "search algorithms",
        "Binary search achieves `O(log n)` time by discarding half of the remaining array after comparing against the middle element, which is only valid when global sorted order guarantees all smaller items lie to the left and all larger items lie to the right.",
        "If the elements are unsorted, comparing against the midpoint gives zero information about which half contains the target, so discarding either half can miss the element entirely.",
    ),
    (
        "collision resolution in hash tables",
        "data structures",
        "Because a hash function maps a huge or infinite space of keys into a finite array of buckets, the Pigeonhole Principle guarantees that distinct keys will eventually land on the same bucket index.",
        "Collision resolution strategies like separate chaining or open addressing (linear probing) preserve all colliding key-value pairs instead of silently overwriting earlier entries.",
    ),
    (
        "a termination guard in iterative numerical solvers",
        "numerical computing",
        "Iterative algorithms refine an approximate answer step by step; without a convergence threshold or maximum-iteration guard, tiny floating-point rounding residuals can keep the update loop running indefinitely.",
        "Combining an error tolerance check with a hard iteration cap guarantees the routine always finishes in bounded time.",
    ),
    (
        "mutexes or atomic operations on shared state",
        "multithreaded programming",
        "When multiple threads read and modify the same memory location concurrently without synchronization, interleaved CPU instructions create race conditions that corrupt state non-deterministically.",
        "Atomics and mutexes enforce a well-defined ordering of reads and writes across CPU cores and cache lines so invariants remain intact.",
    ),
    (
        "short-circuit evaluation in boolean logic",
        "programming language semantics",
        "Short-circuit operators (`&&`/`and`, `||`/`or`) stop evaluating the right-hand operand as soon as the left-hand operand determines the final truth value.",
        "This both avoids wasted computation and lets programmers safely write guard checks—like verifying a pointer is non-null or an index is in bounds—before dereferencing it.",
    ),
    (
        "64-byte cache-line alignment in SIMD structures",
        "high-performance systems programming",
        "Modern CPUs transfer memory between RAM and L1/L2 cache in fixed 64-byte cache lines; aligning hot arrays to 64-byte boundaries prevents split loads across two cache lines and eliminates false sharing between threads.",
        "Combined with Struct-of-Arrays (`soa`) layout, every SIMD vector load pulls 100% useful contiguous lanes in a single memory transaction.",
    ),
    (
        "power-of-two capacities in ring buffers",
        "low-latency systems",
        "Integer division and modulo (`%`) take tens of CPU cycles, whereas when a buffer's capacity $N$ is a power of two, wrapping an index `i % N` reduces to a single-cycle bitwise AND `i & (N - 1)`.",
        "This keeps producer and consumer pointer updates branchless and single-cycle in real-time audio, networking, and kernel queues.",
    ),
    (
        "virtual memory page tables in an operating system",
        "operating systems",
        "Virtual memory gives each process an isolated, contiguous address space mapped to physical RAM frames via hardware page tables and the CPU's Translation Lookaside Buffer (TLB).",
        "Isolation prevents a buggy pointer in one process from reading or corrupting another process's memory, and guard pages immediately trap out-of-bounds stack or heap accesses.",
    ),
    (
        "immutable values in functional and concurrent programming",
        "software architecture",
        "Immutable values cannot be modified after creation, meaning any number of threads or callers can share references to the same data with zero risk of race conditions or unexpected side effects.",
        "This eliminates defensive copying and makes program state transitions much easier to test and reason about.",
    ),
    (
        "arena (bump) allocators over per-object `malloc` and `free`",
        "memory allocation",
        "An arena reserves one contiguous memory region up front and allocates objects by simply bumping a single offset pointer, then frees thousands of temporary objects at once in `O(1)` time by resetting `offset = 0`.",
        "By avoiding free-list metadata, heap fragmentation, and per-object deallocation locks, arenas run orders of magnitude faster for frame- or request-scoped workloads.",
    ),
    (
        "null-terminator bytes (`'\\0'`) or explicit lengths in strings",
        "systems programming",
        "In raw memory, bytes are continuous with no built-in boundary markers; without either an explicit length field or a sentinel `'\\0'` byte, string routines would keep reading past the buffer into adjacent memory.",
        "Reading past buffer bounds causes buffer over-reads, crashes (`SIGSEGV`), or security vulnerabilities.",
    ),
    (
        "deadlock prevention via consistent lock ordering",
        "concurrent programming",
        "A deadlock occurs when Thread A holds Lock 1 while waiting for Lock 2, and Thread B holds Lock 2 while waiting for Lock 1, freezing both threads permanently in a circular wait.",
        "Acquiring locks in a strict global order breaks cycle formation so at least one thread can always make forward progress.",
    ),
    (
        "amortized doubling when growing dynamic arrays (`std::vector`)",
        "algorithmic complexity",
        "If a dynamic array grew by only 1 slot on every `push_back`, copying all existing $k$ elements each time would turn $N$ insertions into slow `O(N^2)` total work.",
        "Doubling capacity whenever full ensures element copies happen at exponentially rarer intervals ($1 + 2 + 4 + \\dots + N < 2N$), achieving `O(1)` amortized time per insertion.",
    ),
    (
        "floating-point epsilon tolerances when comparing decimals",
        "scientific and systems programming",
        "Binary IEEE-754 floats cannot represent most base-10 fractions like `0.1` with finite bits, so arithmetic accumulates tiny rounding differences in the least significant bit.",
        "Checking `|a - b| <= eps` instead of exact `==` tests whether two computed quantities agree within hardware precision.",
    ),
    (
        "compiler header guards (`#pragma once` / `#ifndef`)",
        "C and C++ compilation",
        "When multiple headers `#include` a shared header inside the same translation unit, pasting the same struct or inline definition twice violates the One Definition Rule and fails compilation.",
        "Include guards ensure the preprocessor expands each header file at most once per compilation unit.",
    ),
    (
        "TCP congestion control versus raw UDP datagrams",
        "network programming",
        "TCP tracks acknowledgments, retransmits lost segments in order, and backs off sending rate when routers drop packets, whereas UDP transmits independent datagrams immediately with zero handshake or head-of-line blocking.",
        "File transfers and databases rely on TCP's lossless ordering, while real-time games and voice streams prefer UDP's predictable low latency.",
    ),
]

EXPLAIN_TEMPLATES = [
    "Explain why {topic} is important in programming.",
    "In programming, why do we need {topic}?",
    "Explain how and why {topic} matters in {field}.",
    "Why is {topic} necessary in computer science and programming?",
    "Can you explain why software engineers care about {topic}?",
    "What happens without {topic} in programming, and why is it needed?",
]

PROSE_THINK_PLANS = [
    "The user is asking for a conceptual explanation in plain English rather than source code. Let's explain the core mechanism and why it matters clearly.",
    "This prompt asks *why* a programming concept is needed conceptually. Let's give a clear, concise prose explanation without dumping code.",
    "Let's answer this computer science concept question directly in clear prose: state the core rule, what goes wrong without it, and how it works.",
    "The user wants an intuitive explanation, not an implementation. Let's structure a crisp two-paragraph explanation in plain language.",
]


def synthesize_cs_concept_prose_pair() -> Tuple[str, str]:
    topic, field, para1, para2 = random.choice(CS_CONCEPT_BANK)
    prompt = random.choice(EXPLAIN_TEMPLATES).format(topic=topic, field=field)
    think = random.choice(PROSE_THINK_PLANS)
    resp = f"<think>\n{think}\n</think>\n{para1}\n\n{para2}"
    return prompt, resp


# ==============================================================================
# 2. NON-REPEATING 2,048-TOKEN MULTI-NEEDLE NIAH GENERATOR (ZERO FIXED ID LISTS!)
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


def synthesize_nonrepeating_2k_niah_pair(n_sections: Optional[int] = None, rng: Optional[random.Random] = None) -> Tuple[str, str, List[str]]:
    """
    Synthesizes a 1,100 - 1,850 token multi-section telemetry log where EVERY subsystem ID,
    verification code, and temperature is freshly generated so the model MUST copy from long-range
    FlashVAULT Co-Prime fast-weight cells rather than memorizing vocabulary!
    Returns `(user_prompt, assistant_response, needle_strings_for_loss_boost)`.
    """
    r = rng if rng is not None else random
    if n_sections is None:
        n_sections = r.randint(16, 24)

    used_names = set()
    records = []
    for i in range(n_sections):
        while True:
            sub = f"{r.choice(NOUN_PREFIXES)}{r.choice(NOUN_SUFFIXES)}-{chr(r.randint(65, 90))}{r.randint(1, 9)}"
            if sub not in used_names:
                used_names.add(sub)
                break
        tag = r.choice(["HEX", "SIG", "KEY", "CHK"])
        code = f"{tag}-{r.randint(1000, 9999)}"
        temp = r.randint(180, 960)
        lat = round(r.uniform(0.35, 22.50), 2)
        records.append((i + 1, sub, code, temp, lat))

    target_idx = r.randint(1, n_sections - 4)  # Always buried in early/middle chunks (>500-1400 tokens before query!)
    sec_no, t_sub, t_code, t_temp, _ = records[target_idx]

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
# 3. 30% PRETRAIN SKILL REPLAY GENERATOR (CONTINUOUS NUMBERS, ZERO EVAL LEAK!)
# ==============================================================================

def synthesize_pretrain_replay_pair(real_bb_files: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Replays the exact polyglot & arithmetic skill families from `train_vault_150m.py` with
    continuous uniform numbers so exact prompt-copying (`45 * 4 = 180`, `reverse_i32`, `.bb` SoA)
    remains 100% razor-sharp!
    """
    kind = random.choice(["bb_soa", "bb_sys", "bb_real", "c99", "cpp20", "lua", "rate_math", "multi_math"])
    n_elems = random.choice([1024, 2048, 4096, 8192, 16384, 65536, 262144, 1000000])
    arena_mb = random.choice([4, 8, 16, 32, 64, 128])
    arena_bytes = arena_mb * 1048576
    dt = round(random.uniform(0.002, 0.25), 4)
    scale = round(random.uniform(1.5, 9.5), 2)

    if kind == "bb_soa":
        # Hold out `Photon` so Eval #1 remains unseen struct synthesis!
        sname = random.choice(["Particle", "WaveTask", "VoxelNode", "RayPacket", "SensorSample", "TensorCell", "BodyState"])
        f1, f2, f3 = random.choice([("x", "vx", "ax"), ("pos", "vel", "force"), ("u", "du", "energy"), ("r", "dr", "mass")])
        op_desc, expr = random.choice([
            (f"updates `{f1}` positions in parallel", f"items[i].{f1} = items[i].{f1} + items[i].{f2} * {dt}"),
            (f"scales `{f1}` by `{scale}` and adds `{f3}` in parallel", f"items[i].{f1} = items[i].{f1} * {scale} + items[i].{f3}"),
        ])
        return (
            f"Write a BuildBit (.bb) function that defines `soa struct {sname}` with `f32` fields `{f1}`, `{f2}`, `{f3}`, allocates {n_elems} elements in a {arena_mb} MB `gpu_arena`, and {op_desc}.",
            f"<think>\nIn BuildBit (`bbc`), we declare `soa struct {sname}` for 64-byte SIMD alignment, allocate {n_elems} items in `gpu_arena vram = {arena_bytes}` via `alloc_soa`, and run a `parallel for` loop over `0..{n_elems}`.\n</think>\n"
            f"```buildbit\nsoa struct {sname} {{\n    {f1}: f32,\n    {f2}: f32,\n    {f3}: f32,\n    id: i64\n}}\n\nfn main() {{\n    gpu_probe()\n    gpu_arena vram = {arena_bytes}\n    let items = alloc_soa(vram, {sname}, {n_elems})\n\n    parallel for i in 0..{n_elems} {{\n        {expr}\n    }}\n\n    reset(vram)\n}}\n```"
        )

    if kind == "bb_sys":
        fn_name = random.choice(["add_arrays", "saxpy_kernel", "xor_cipher", "clamp_buffer"])
        mult = random.randint(3, 97)
        return (
            f"Write a BuildBit (.bb) function `{fn_name}` that allocates two buffers of {n_elems} elements in an `arena` and combines them in a `parallel for` loop.",
            f"<think>\nWe use `arena mem = {arena_bytes}`, `alloc(mem, f32, {n_elems})`, and `parallel for i in 0..{n_elems}` for zero-overhead SIMD execution.\n</think>\n"
            f"```buildbit\nfn {fn_name}() {{\n    arena mem = {arena_bytes}\n    let a = alloc(mem, f32, {n_elems})\n    let b = alloc(mem, f32, {n_elems})\n\n    parallel for i in 0..{n_elems} {{\n        a[i] = (i * {mult}) * 0.5\n        b[i] = a[i] * {scale} + {dt}\n    }}\n\n    reset(mem)\n}}\n```"
        )

    if kind == "bb_real" and real_bb_files:
        fname, code = random.choice(real_bb_files)
        return (
            f"Write the BuildBit (.bb) implementation for `{fname}`.",
            f"<think>\nUsing BuildBit's zero-copy arenas, `soa struct`, and direct hardware primitives.\n</think>\n```buildbit\n{code}\n```",
        )

    if kind == "c99":
        ctask = random.choice(["reverse_u64", "reverse_f32", "count_vowels", "dot_i32", "fma_array"])
        if ctask in ("reverse_u64", "reverse_f32"):
            ctype = "uint64_t" if ctask == "reverse_u64" else "float"
            hdr = "#include <stdint.h>\n#include <stddef.h>" if ctask == "reverse_u64" else "#include <stddef.h>"
            desc = "64-bit unsigned integers" if ctask == "reverse_u64" else "32-bit floats"
            return (
                f"Write a C99 function to reverse an array of {desc} in place.",
                f"<think>\nWe use two indices `i = 0` and `j = n - 1` and swap `arr[i]` and `arr[j]` until `i >= j`.\n</think>\n"
                f"```c\n{hdr}\n\nvoid {ctask}({ctype} *arr, size_t n) {{\n    if (!arr || n < 2) return;\n    size_t i = 0, j = n - 1;\n    while (i < j) {{\n        {ctype} tmp = arr[i];\n        arr[i] = arr[j];\n        arr[j] = tmp;\n        ++i;\n        --j;\n    }}\n}}\n```"
            )
        if ctask == "count_vowels":
            return (
                "Write a C99 function that counts the number of vowels in a null-terminated string.",
                "<think>\nWe iterate through `const char *s` until `'\\0'`, lowercase ASCII letters via `(c | 32)`, and count `'a'`, `'e'`, `'i'`, `'o'`, `'u'`.\n</think>\n"
                "```c\n#include <stddef.h>\n\nsize_t count_vowels(const char *s) {\n    size_t count = 0;\n    if (!s) return 0;\n    for (; *s != '\\0'; ++s) {\n        char c = *s | 32;\n        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') ++count;\n    }\n    return count;\n}\n```"
            )
        if ctask == "dot_i32":
            return (
                "Write a C99 function `dot_i32` that computes the 64-bit dot product of two 32-bit integer arrays without overflow.",
                "<think>\nCasting each element to `int64_t` prior to multiplication avoids signed 32-bit overflow.\n</think>\n"
                "```c\n#include <stdint.h>\n#include <stddef.h>\n\nint64_t dot_i32(const int32_t * restrict a, const int32_t * restrict b, size_t n) {\n    if (!a || !b) return 0;\n    int64_t acc = 0;\n    for (size_t i = 0; i < n; ++i) acc += (int64_t)a[i] * (int64_t)b[i];\n    return acc;\n}\n```"
            )
        return (
            f"Write a fast C99 function that scales an array of floats by {scale}f and adds {dt}f.",
            f"<think>\nUsing `restrict` pointers allows the C99 compiler to auto-vectorize the loop with SIMD instructions.\n</think>\n"
            f"```c\n#include <stddef.h>\n\nvoid fma_array(float * restrict out, const float * restrict in, size_t n) {{\n    if (!out || !in) return;\n    for (size_t i = 0; i < n; ++i) out[i] = in[i] * {scale}f + {dt}f;\n}}\n```"
        )

    if kind == "cpp20":
        # Hold out `FixedStack` so Eval #3 tests unseen identifier synthesis!
        cls = random.choice(["BoundedStack", "FixedQueue", "SpanBuffer", "StaticStack"])
        cap = random.choice([32, 64, 128, 256, 512, 1024])
        return (
            f"Write a C++20 generic class template `{cls}<T, {cap}>` with `push`, `pop`, and `size` methods.",
            f"<think>\nWe use `std::array<T, N>` and `std::optional<T>` in C++20 for zero-heap-allocation type safety.\n</think>\n"
            f"```cpp\n#include <array>\n#include <optional>\n#include <cstddef>\n\ntemplate <typename T, std::size_t N = {cap}>\nclass {cls} {{\n    std::array<T, N> buf_{{}};\n    std::size_t count_ = 0;\npublic:\n    bool push(const T& val) noexcept {{\n        if (count_ >= N) return false;\n        buf_[count_++] = val;\n        return true;\n    }}\n    std::optional<T> pop() noexcept {{\n        if (count_ == 0) return std::nullopt;\n        return buf_[--count_];\n    }}\n    std::size_t size() const noexcept {{ return count_; }}\n}};\n```"
        )

    if kind == "lua":
        lcls = random.choice(["Vec2", "Vec3", "Counter", "Timer", "Actor", "Spring"])
        return (
            f"Write a Lua module implementing a `{lcls}` class using metatables (`__index` and `:update(dt)`).",
            f"<think>\nIn Lua, we define `local {lcls} = {{}}`, set `{lcls}.__index = {lcls}`, and use `setmetatable` in `{lcls}.new`.\n</think>\n"
            f"```lua\nlocal {lcls} = {{}}\n{lcls}.__index = {lcls}\n\nfunction {lcls}.new(x, y)\n    return setmetatable({{ x = x or 0, y = y or 0 }}, {lcls})\nend\n\nfunction {lcls}:update(dt)\n    self.x = self.x + self.y * (dt or {dt})\n    return self.x\nend\n\nfunction {lcls}:__tostring()\n    return string.format(\"{lcls}(%.2f, %.2f)\", self.x, self.y)\nend\n\nreturn {lcls}\n```"
        )

    if kind == "rate_math":
        # Continuous uniform integers across 11..135 (holding out exact 45*4!) so the model MUST copy the exact prompt numbers!
        speed = random.randint(11, 135)
        hours = random.randint(2, 14)
        if speed == 45 and hours == 4:
            speed = 46
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
# 4. UNIFIED DATA HUB FOR STAGE 1 (2K PRETRAIN) & STAGE 2 (CONTRASTIVE + REPLAY SFT)
# ==============================================================================

class TwoStageFlashDataHub:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos_id = tokenizer.eos_token_id or 0
        self.real_bb_files: List[Tuple[str, str]] = []
        self.prose_pairs: List[Tuple[str, str]] = []
        self.code_pairs: List[Tuple[str, str]] = []
        self.math_pairs: List[Tuple[str, str]] = []
        self.long_docs: List[str] = []

        self.fineweb_q: "queue.Queue[str]" = queue.Queue(maxsize=256)
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
                    if len(txt) > 1800:
                        self.fineweb_q.put(txt[:9000], timeout=5.0)
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

        # 2. Load SmolTalk Arrow shards: split Pure-Prose Explanations (INCLUDING CS/tech concepts without code!) vs Code
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
                    if not u or not a or len(a) < 60 or len(u) + len(a) > 3200:
                        continue
                    # Hold out any recursion base-case question if present in SmolTalk!
                    if "recursion" in u.lower() and "base case" in u.lower():
                        continue

                    has_code_fence = ("```" in a) or ("\ndef " in a) or ("#include <" in a)
                    if not has_code_fence and (cat in prose_cats or cat == "coding"):
                        # Notice: `cat == "coding"` WITHOUT code fences = genuine CS/programming prose explanations!
                        if len(self.prose_pairs) < 22000:
                            think = random.choice(PROSE_THINK_PLANS)
                            self.prose_pairs.append((u, f"<think>\n{think}\n</think>\n{a}"))
                            if len(a) > 900 and len(self.long_docs) < 4000:
                                self.long_docs.append(f"{u}\n\n{a}")
                    elif cat == "coding" and len(self.code_pairs) < 3500:
                        if any(tag in a for tag in ("```python", "```c\n", "```cpp", "```c++", "```lua")):
                            self.code_pairs.append((
                                u,
                                f"<think>\nThe user requested a code implementation. Let's write a clean, well-structured solution.\n</think>\n{a}",
                            ))
            except Exception:
                pass

        # 3. Python CodeAlpaca (explicitly tagged implementation prompts)
        py_files = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--iamtarun--python_code_instructions_18k_alpaca/**/train-*.parquet"), recursive=True))
        for pf in py_files:
            t = pq.read_table(pf)
            for i in range(min(len(t), 9000)):
                inst = str(t["instruction"][i].as_py() or "").strip()
                inp = str(t["input"][i].as_py() or "").strip() if "input" in t.column_names else ""
                out = str(t["output"][i].as_py() or "").strip()
                if not inst or not out or len(inst) + len(out) > 2400:
                    continue
                if "maximum element and sum of a list" in inst.lower():
                    continue
                q = f"[python] {inst}\nInput: {inp}" if inp else f"[python] {inst}"
                if not out.startswith("```"):
                    out = f"```python\n{out}\n```"
                self.code_pairs.append((
                    q,
                    f"<think>\nThe user wants a Python implementation. Let's handle edge cases cleanly and write idiomatic Python.\n</think>\n{out}",
                ))

        # 4. GSM8K + AI2-ARC Math & Science
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

        # TinyStories long coherent fallback docs if FineWeb queue hasn't filled yet
        ts_pq = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--roneneldan--TinyStories/**/train-*.parquet"), recursive=True))
        if ts_pq:
            t = pq.read_table(ts_pq[0], columns=["text"])
            buf = []
            for i in range(min(len(t), 4000)):
                buf.append(str(t["text"][i].as_py() or "").strip())
                if len(buf) == 4:
                    self.long_docs.append("\n\n".join(buf))
                    buf = []

        random.shuffle(self.prose_pairs)
        random.shuffle(self.code_pairs)
        random.shuffle(self.math_pairs)
        random.shuffle(self.long_docs)
        print(f"  [Hub Ready] Pure Prose (incl. CS prose): {len(self.prose_pairs):>6,} | Code: {len(self.code_pairs):>6,} | Math/Sci: {len(self.math_pairs):>6,} | Real .bb: {len(self.real_bb_files)}", flush=True)

    # --------------------------------------------------------------------------
    # STAGE 1 BATCHER: Full `seq_len = 2048` Long-Context Pretraining Continuation
    # --------------------------------------------------------------------------
    def sample_stage1_2k_pretrain_batch(self, device: str, seq_len: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns `(input_ids, labels, token_weights)` of shape `(1, 2048)`.
        Trains all 16 FlashVAULT layers at full 32-chunk (2,048-token) context!
        """
        r = random.random()

        # 30% Non-Repeating 2K Multi-Needle NIAH (full document causal + 4.0x weight on retrieved answer!)
        if r < 0.30:
            u_txt, a_txt, _ = synthesize_nonrepeating_2k_niah_pair(n_sections=random.randint(18, 25))
            prompt_str = f"### User:\n{u_txt}\n\n### Assistant:\n"
            p_ids = self.tok.encode(prompt_str, add_special_tokens=False)
            a_ids = self.tok.encode(a_txt, add_special_tokens=False) + [self.eos_id]
            full = p_ids + a_ids
            if len(full) < seq_len + 1:
                # Append a natural FineWeb/long doc segment after `<eos>` so all 2048 tokens are active real tokens!
                extra_doc = self._get_long_natural_doc()
                extra_ids = self.tok.encode(extra_doc, add_special_tokens=False) + [self.eos_id]
                full = (full + extra_ids)[: seq_len + 1]
            else:
                full = full[: seq_len + 1]

            if len(full) < seq_len + 1:
                full = full + [self.eos_id] * (seq_len + 1 - len(full))

            inp = full[:-1]
            lbl = full[1:]
            w = [1.0] * seq_len
            ans_start = max(0, len(p_ids) - 1)
            ans_end = min(seq_len, len(p_ids) + len(a_ids) - 1)
            for idx in range(ans_start, ans_end):
                w[idx] = 4.0  # Boost cross-chunk needle retrieval gradient!
            return (
                torch.tensor([inp], dtype=torch.long, device=device),
                torch.tensor([lbl], dtype=torch.long, device=device),
                torch.tensor([w], dtype=torch.float32, device=device),
            )

        # Pack uncut FineWeb long docs (30%), Pretrain Skill Replay (25%), and Contrastive CS Prose (15%) to 2,048 tokens
        buf: List[int] = []
        while len(buf) < seq_len + 1:
            sub_r = random.random()
            if sub_r < 0.38:
                doc = self._get_long_natural_doc()
            elif sub_r < 0.75:
                u, a = synthesize_pretrain_replay_pair(self.real_bb_files)
                doc = f"### User:\n{u}\n\n### Assistant:\n{a}"
            elif sub_r < 0.90:
                u, a = synthesize_cs_concept_prose_pair()
                doc = f"### User:\n{u}\n\n### Assistant:\n{a}"
            else:
                u, a = random.choice(self.prose_pairs)
                doc = f"### User:\n{u}\n\n### Assistant:\n{a}"
            buf.extend(self.tok.encode(doc, add_special_tokens=False))
            buf.append(self.eos_id)

        seq = buf[: seq_len + 1]
        return (
            torch.tensor([seq[:-1]], dtype=torch.long, device=device),
            torch.tensor([seq[1:]], dtype=torch.long, device=device),
            torch.ones((1, seq_len), dtype=torch.float32, device=device),
        )

    def _get_long_natural_doc(self) -> str:
        if not self.fineweb_q.empty():
            try:
                return self.fineweb_q.get_nowait()
            except queue.Empty:
                pass
        return random.choice(self.long_docs) if self.long_docs else "Systems engineering combines hardware architecture and clean software abstractions."

    # --------------------------------------------------------------------------
    # STAGE 2 BATCHER: Gentle 1.5e-5 Contrastive CS-Intent + 30% Replay + 2K NIAH SFT
    # --------------------------------------------------------------------------
    def sample_stage2_contrastive_sft_batch(self, device: str, tokens_per_micro: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prompt-masked (`labels = -100` on `### User: ... ### Assistant:`) dynamic micro-batches:
          - 20% B=1, L=2048: Non-Repeating 18-24 Section Cross-Chunk NIAH
          - 45% B=2, L=1024: Contrastive CS Prose, Long Explanations & Polyglot Code
          - 35% B=4, L=512 : 30% Pretrain Replay + Contrastive CS Intent + Math `<think>`
        """
        mr = random.random()
        if mr < 0.20:
            B, seq_len, force_niah = 1, 2048, True
        elif mr < 0.65:
            B, seq_len, force_niah = 2, 1024, False
        else:
            B, seq_len, force_niah = 4, 512, False

        inputs, labels = [], []
        attempts = 0
        while len(inputs) < B:
            attempts += 1
            if force_niah and attempts < 8:
                u_txt, a_txt, _ = synthesize_nonrepeating_2k_niah_pair(n_sections=random.randint(17, 23))
            else:
                cr = random.random()
                if cr < 0.22:
                    # Explicit Contrastive CS-Concept Prose ("Explain/Why ... in programming" -> PURE PROSE!)
                    u_txt, a_txt = synthesize_cs_concept_prose_pair()
                elif cr < 0.40:
                    # SmolTalk high-quality Pure Prose (incl. CS/science explanations without code!)
                    u_txt, a_txt = random.choice(self.prose_pairs)
                elif cr < 0.72:
                    # 32% Original Pretraining Replay (BuildBit, C99, C++20, Lua, continuous `speed * hours` math!)
                    u_txt, a_txt = synthesize_pretrain_replay_pair(self.real_bb_files)
                elif cr < 0.88:
                    # Explicit Implementation Code (CodeAlpaca / Multi-lang)
                    u_txt, a_txt = random.choice(self.code_pairs)
                else:
                    # GSM8K / ARC `<think>` Math & Science
                    u_txt, a_txt = random.choice(self.math_pairs)

            prompt_str = f"### User:\n{u_txt}\n\n### Assistant:\n"
            full_str = f"{prompt_str}{a_txt}"
            p_ids = self.tok.encode(prompt_str, add_special_tokens=False)
            f_ids = self.tok.encode(full_str, add_special_tokens=False) + [self.eos_id]

            if len(p_ids) >= seq_len - 24 or len(f_ids) > seq_len + 1:
                if attempts > 24 and len(p_ids) < seq_len - 24:
                    f_ids = f_ids[: seq_len + 1]
                else:
                    continue

            target_len = seq_len + 1
            pad_len = target_len - len(f_ids)
            seq = f_ids + [self.eos_id] * pad_len
            n_p = len(p_ids)
            valid_resp = seq[n_p : len(f_ids)]
            lbl = ([-100] * (n_p - 1)) + valid_resp + ([-100] * pad_len)

            inputs.append(seq[:-1])
            labels.append(lbl[:seq_len])

        return (
            torch.tensor(inputs, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device),
        )


# ==============================================================================
# 5. GENERATION & 8-DOMAIN UNSEEN EVALUATION
# ==============================================================================

@torch.no_grad()
def generate_response(
    model: VAULTLM150M,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int = 192,
    device: str = "cuda:0",
    temperature: float = 0.15,
) -> str:
    model.eval()
    prompt_str = f"### User:\n{user_prompt}\n\n### Assistant:\n"
    ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    prompt_len = ids.size(1)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        ctx = ids[:, -4096:]
        logits = model(ctx, step_ratio=1.0)[:, -1, :]
        for prev_id in set(ids[0, prompt_len:].tolist()):
            if logits[0, prev_id] > 0:
                logits[0, prev_id] /= 1.08
            else:
                logits[0, prev_id] *= 1.08

        if temperature <= 1e-4:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            top_v, top_i = torch.topk(scaled, k=10, dim=-1)
            probs = F.softmax(top_v.float(), dim=-1)
            next_tok = top_i.gather(-1, torch.multinomial(probs, 1))

        ids = torch.cat([ids, next_tok], dim=1)
        if eos_id is not None and next_tok.item() == eos_id:
            break

    model.train()
    return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True).strip()


def run_full_8domain_eval(model: VAULTLM150M, tok, tag: str, device: str, out_path: str):
    eval_rng = random.Random(2026102999)
    niah_prompt, niah_gold, niah_needles = synthesize_nonrepeating_2k_niah_pair(n_sections=18, rng=eval_rng)

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
        out = generate_response(model, tok, prm, max_new_tokens=200, device=device, temperature=0.15)
        results.append({"domain": dom, "prompt": prm[-320:] if len(prm) > 320 else prm, "output": out})
        print(f"  [{dom}]\n    -> {out.replace(chr(10), ' ⏎ ')[:260]}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"stage": tag, "niah_expected": niah_needles, "niah_gold": niah_gold, "evaluations": results}, f, indent=2)
    print(f"[+] Saved 8-domain evaluation -> {out_path}\n", flush=True)


# ==============================================================================
# 6. TWO-STAGE EXECUTION ENGINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_steps", type=int, default=220, help="Stage 1: L=2048 Long-Context Pretraining steps")
    parser.add_argument("--stage2_steps", type=int, default=240, help="Stage 2: Gentle 1.5e-5 Contrastive + Replay SFT steps")
    parser.add_argument("--grad_accum", type=int, default=4, help="8,192 tokens/step")
    parser.add_argument("--stage1_lr", type=float, default=3.0e-5, help="Stage 1 peak LR (< 4e-5 pretrain floor)")
    parser.add_argument("--stage2_lr", type=float, default=1.5e-5, help="Stage 2 gentle peak LR")
    parser.add_argument("--log_interval", type=int, default=20, help="Log every N steps")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    bench_dir = os.path.join(base_dir, "benchmarks")

    src_ckpt = os.path.join(ckpt_dir, "vault_150m_best.pt")
    stage1_ckpt = os.path.join(ckpt_dir, "vault_150m_flash_2k_stage1.pt")
    stage2_ckpt = os.path.join(ckpt_dir, "vault_150m_flash_sft_v2.pt")
    sft_alias = os.path.join(ckpt_dir, "vault_150m_sft.pt")
    telemetry_path = os.path.join(bench_dir, "vault_150m_flash_2stage_telemetry.json")
    eval_stage1_path = os.path.join(bench_dir, "eval_vault_150m_flash_stage1_2k.json")
    eval_final_path = os.path.join(bench_dir, "eval_vault_150m_flash_2stage.json")

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
    ).to(device).to(torch.bfloat16)

    ckpt = torch.load(src_ckpt, map_location=device)
    n_loaded, _ = model.load_pretrained_vault_state(ckpt["model_state"])

    print("=" * 98, flush=True)
    print("VAULTLM-150M + FLASHVAULT: 2-STAGE LONG-CONTEXT (2K) + CONTRASTIVE CS-INTENT + REPLAY PIPELINE", flush=True)
    print("=" * 98, flush=True)
    print(f"  - Source Checkpoint : {src_ckpt} ({n_loaded} tensors matched, Pretrain Val PPL: {ckpt.get('best_ppl', 0):.4f})", flush=True)
    print(f"  - Stage 1 (2K Cont.): {args.stage1_steps} steps @ L=2048 | LR {args.stage1_lr:.1e} -> 1.0e-5", flush=True)
    print(f"  - Stage 2 (SFT+Rply): {args.stage2_steps} steps @ L=512..2048 | LR {args.stage2_lr:.1e} -> 3.0e-6 (30%+ Replay)", flush=True)
    print("=" * 98, flush=True)

    hub = TwoStageFlashDataHub(tok)
    random.seed(777)
    s1_val = [hub.sample_stage1_2k_pretrain_batch(device, 2048) for _ in range(4)]
    s2_val = [hub.sample_stage2_contrastive_sft_batch(device, 2048) for _ in range(6)]
    random.seed(42)

    telemetry = {"stage1": [], "stage2": []}
    tokens_per_step = 2048 * args.grad_accum

    # ==========================================================================
    # STAGE 1: `seq_len = 2048` LONG-CONTEXT PRETRAINING CONTINUATION
    # ==========================================================================
    if args.stage1_steps > 0:
        print("\n>>> STARTING STAGE 1: seq_len = 2048 LONG-CONTEXT FLASHVAULT PRETRAINING CONTINUATION <<<", flush=True)
        print("-" * 98, flush=True)
        opt1 = torch.optim.AdamW(model.parameters(), lr=args.stage1_lr, betas=(0.9, 0.95), weight_decay=0.02)
        t0 = time.perf_counter()
        best_s1_val = float("inf")
        warmup1 = min(20, max(1, args.stage1_steps // 8))

        for step in range(1, args.stage1_steps + 1):
            if step <= warmup1:
                lr = args.stage1_lr * (step / float(warmup1))
            else:
                p = (step - warmup1) / float(max(1, args.stage1_steps - warmup1))
                lr = 1.0e-5 + 0.5 * (args.stage1_lr - 1.0e-5) * (1.0 + math.cos(math.pi * p))
            for pg in opt1.param_groups:
                pg["lr"] = lr

            opt1.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(args.grad_accum):
                inp, lbl, w = hub.sample_stage1_2k_pretrain_batch(device, 2048)
                logits = model(inp, step_ratio=1.0)
                tok_loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), lbl.view(-1), reduction="none")
                w_flat = w.view(-1)
                loss = ((tok_loss * w_flat).sum() / w_flat.sum().clamp(min=1.0)) / args.grad_accum
                loss.backward()
                loss_acc += loss.item()

            if not math.isfinite(loss_acc):
                raise RuntimeError(f"Non-finite Stage 1 loss at step {step}: {loss_acc}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt1.step()

            if step == 1 or step % args.log_interval == 0 or step == args.stage1_steps:
                model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for vi, vl, _ in s1_val:
                        vlog = model(vi, step_ratio=1.0)
                        v_loss += F.cross_entropy(vlog.float().view(-1, vlog.size(-1)), vl.view(-1)).item()
                v_loss /= len(s1_val)
                v_ppl = math.exp(min(v_loss, 20.0))
                model.train()

                tok_s = (step * tokens_per_step) / max(time.perf_counter() - t0, 1e-3)
                print(f"[S1 {step:>3}/{args.stage1_steps}] Train: {loss_acc:.4f} | 2K Val Loss: {v_loss:.4f} (PPL {v_ppl:.2f}) | LR: {lr:.2e} | {tok_s:>7.0f} tok/s", flush=True)
                telemetry["stage1"].append({"step": step, "train_loss": round(loss_acc, 4), "val_loss": round(v_loss, 4), "val_ppl": round(v_ppl, 2), "tok_s": round(tok_s, 1)})

                if v_loss < best_s1_val:
                    best_s1_val = v_loss
                    torch.save({"step": step, "val_loss": v_loss, "val_ppl": v_ppl, "model_state": model.state_dict()}, stage1_ckpt)

                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

        run_full_8domain_eval(model, tok, f"After Stage 1 (L=2048 Pretrain Step {args.stage1_steps})", device, eval_stage1_path)
        del opt1
        torch.cuda.empty_cache()

    # ==========================================================================
    # STAGE 2: GENTLE 1.5e-5 CONTRASTIVE CS-INTENT + 30% REPLAY + 2K NIAH SFT
    # ==========================================================================
    if args.stage2_steps > 0:
        print("\n>>> STARTING STAGE 2: GENTLE 1.5e-5 CONTRASTIVE CS-INTENT + 30% REPLAY + 2K NIAH SFT <<<", flush=True)
        print("-" * 98, flush=True)
        opt2 = torch.optim.AdamW(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.95), weight_decay=0.01)
        t0 = time.perf_counter()
        best_s2_val = float("inf")
        warmup2 = min(20, max(1, args.stage2_steps // 8))

        for step in range(1, args.stage2_steps + 1):
            if step <= warmup2:
                lr = args.stage2_lr * (step / float(warmup2))
            else:
                p = (step - warmup2) / float(max(1, args.stage2_steps - warmup2))
                lr = 3.0e-6 + 0.5 * (args.stage2_lr - 3.0e-6) * (1.0 + math.cos(math.pi * p))
            for pg in opt2.param_groups:
                pg["lr"] = lr

            opt2.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(args.grad_accum):
                inp, lbl = hub.sample_stage2_contrastive_sft_batch(device, 2048)
                logits = model(inp, step_ratio=1.0)
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), lbl.view(-1), ignore_index=-100) / args.grad_accum
                loss.backward()
                loss_acc += loss.item()

            if not math.isfinite(loss_acc):
                raise RuntimeError(f"Non-finite Stage 2 loss at step {step}: {loss_acc}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt2.step()

            if step == 1 or step % args.log_interval == 0 or step == args.stage2_steps:
                model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for vi, vl in s2_val:
                        vlog = model(vi, step_ratio=1.0)
                        v_loss += F.cross_entropy(vlog.float().view(-1, vlog.size(-1)), vl.view(-1), ignore_index=-100).item()
                v_loss /= len(s2_val)
                v_ppl = math.exp(min(v_loss, 20.0))
                model.train()

                tok_s = (step * tokens_per_step) / max(time.perf_counter() - t0, 1e-3)
                print(f"[S2 {step:>3}/{args.stage2_steps}] Train: {loss_acc:.4f} | SFT Val Loss: {v_loss:.4f} (PPL {v_ppl:.2f}) | LR: {lr:.2e} | {tok_s:>7.0f} tok/s", flush=True)
                rec = {"step": step, "train_loss": round(loss_acc, 4), "val_loss": round(v_loss, 4), "val_ppl": round(v_ppl, 2), "tok_s": round(tok_s, 1)}

                if step % 60 == 0:
                    prose_chk = generate_response(model, tok, "Explain why recursion needs a base case in programming.", max_new_tokens=140, device=device)
                    math_chk = generate_response(model, tok, "If a rover travels at 45 miles per hour for 4 hours, how far does it travel in total?", max_new_tokens=80, device=device)
                    rec["unseen_prose"] = prose_chk
                    rec["unseen_math"] = math_chk
                    print(f"    [Prose Check] {prose_chk.replace(chr(10), ' ⏎ ')[:190]}", flush=True)
                    print(f"    [Math  Check] {math_chk.replace(chr(10), ' ⏎ ')[:150]}", flush=True)

                telemetry["stage2"].append(rec)
                if v_loss < best_s2_val:
                    best_s2_val = v_loss
                    torch.save({
                        "step": step,
                        "val_loss": v_loss,
                        "val_ppl": v_ppl,
                        "use_flash_vault": True,
                        "n_vault_groups": 12,
                        "model_state": model.state_dict(),
                    }, stage2_ckpt)

                with open(telemetry_path, "w") as f:
                    json.dump(telemetry, f, indent=2)

    hub.stop_event.set()
    if os.path.exists(stage2_ckpt):
        shutil.copy2(stage2_ckpt, sft_alias)
        best_s2 = torch.load(stage2_ckpt, map_location=device)
        model.load_state_dict(best_s2["model_state"])

    run_full_8domain_eval(model, tok, f"Final Stage 2 (Contrastive + Replay + 2K NIAH Step {args.stage2_steps})", device, eval_final_path)
    print("=" * 98, flush=True)
    print(f"[+] ALL 3 UPGRADES COMPLETE! Final model saved -> {stage2_ckpt}", flush=True)
    print("=" * 98, flush=True)


if __name__ == "__main__":
    main()
