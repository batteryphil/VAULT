# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2 Data Procurement, Decontamination & Pre-Tokenization Pipeline
# Builds 100% local NVMe uint16 memmap datasets (`data_v2/`) at full seq_len = 2048:
#   - Stage 1 Pretraining (`train_tokens.bin`, `train_mask.bin`, `val_tokens.bin`, `val_mask.bin`)
#   - Stage 2 Clean ChatML SFT (`sft_train_tokens.bin`, `sft_train_mask.bin`, `sft_val_tokens.bin`, `sft_val_mask.bin`)
#
# Strict IP Scope: Uses ONLY Philip Bartsch's owned repos (`vault`, `buildbit`, `buildbit-os`)
# + verified public HuggingFace datasets downloaded to local NVMe.

import os
import re
import glob
import json
import math
import time
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from transformers import AutoTokenizer


SEQ_LEN = 2048
ROW_LEN = SEQ_LEN + 1  # 2049 tokens -> x = row[:2048], y = row[1:2049]

# Loss weight codes stored as uint8 (actual float weight = code / 10.0):
W_IGNORE = 0    # 0.0x (padding or masked SFT prompt)
W_PROMPT = 3    # 0.3x (pretrain user prompt / haystack context)
W_PROSE = 10    # 1.0x (standard pretrain textbook / web prose / raw code)
W_ASSIST = 15   # 1.5x (assistant explanation / code / <think> derivation)
W_NEEDLE = 30   # 3.0x (exact retrieved needle values, final math answers, <|im_end|>)


# ==============================================================================
# 1. STRICT DECONTAMINATION & BENCHMARK HOLDOUT FILTERS
# ==============================================================================

POISON_CRITIQUE_PHRASES = (
    "given code",
    "provided code",
    "revised code",
    "revised version",
    "corrected code",
    "fixed code",
    "updated code",
    "your code",
    "original code",
    "above code",
    "snippet above",
    "code you provided",
    "here's the modified",
    "here is the modified",
    "as an ai language model",
    "as a large language model",
    "i cannot fulfill",
    "i'm sorry, but i can't",
)

# Exact unseen evaluation targets held out across ALL QA / Instruction / Code / Math / NIAH data
HOLDOUT_SUBSTRINGS = (
    "soa struct photon",
    "reverse an array of 32-bit integers",
    "fixedstack<",
    "maximum element and sum of a list",
    "rover travels at 45",
    "45 miles per hour for 4 hours",
    "why recursion needs a base case",
    "why does recursion need a base case",
    "recursion need a base case",
)

# Unseen NIAH evaluation subsystem prefixes/nouns reserved exclusively for benchmark seed
HOLDOUT_NIAH_PREFIXES = {
    "ShieldPod", "CryoLock", "WarpCore", "IonThruster", "PlasmaGrid", "StarTracker",
    "BeaconMux", "GravSensor", "SolarWing", "HullRadar", "ThermalSink", "DockClamp",
}


def is_clean_pair(prompt: str, response: str) -> Tuple[bool, str]:
    if not prompt or not response or len(prompt.strip()) < 8 or len(response.strip()) < 12:
        return False, "too_short"
    rl = response.lower()
    pl = prompt.lower()
    for p in POISON_CRITIQUE_PHRASES:
        if p in rl:
            return False, "code_review_poison"
    combined = pl + "\n" + rl
    for h in HOLDOUT_SUBSTRINGS:
        if h in combined:
            return False, "benchmark_holdout"
    if "particle" in combined and "metatable" in combined and "update(dt)" in combined:
        return False, "benchmark_holdout"
    return True, "ok"


# ==============================================================================
# 2. PROCEDURAL CLEAN GENERATORS (POLYGLOT SYSTEMS, CONCEPTUAL PROSE, 2K NIAH)
# ==============================================================================

CS_CONCEPT_TOPICS = [
    (
        "an exit condition in a `while` loop",
        "An exit condition defines the exact state at which a `while` loop stops iterating and transfers control to the following statement. Without a reachable exit condition whose variables move toward termination on each pass, the loop guard stays true indefinitely, trapping the thread in an infinite loop that consumes CPU cycles without ever returning a result.",
    ),
    (
        "a stopping depth in nested function calls",
        "Each active function invocation pushes a stack frame holding return addresses, saved registers, and local variables onto a finite call stack in RAM. Without an explicit stopping condition that returns a direct value at a trivial input size, nested calls keep allocating frames until the thread exhausts its stack region and faults with a stack overflow.",
    ),
    (
        "null-child boundary checks in binary tree traversals",
        "Tree algorithms descend from parent nodes to left and right child pointers step by step, so checking for a `null` or `None` pointer establishes the leaf boundary where downward traversal halts. Omitting that check causes the routine to dereference an invalid null address, crashing immediately instead of unwinding cleanly back up the tree.",
    ),
    (
        "a base case in mathematical induction and divide-and-conquer decompositions",
        "Divide-and-conquer algorithms repeatedly split an instance of size `N` into smaller subproblems, which only terminates when subproblems of size `0` or `1` are solved directly without further splitting. That elementary direct solution anchors the entire chain of reductions so every parent level can combine concrete values.",
    ),
    (
        "stack allocation versus heap allocation",
        "Stack memory manages short-lived local variables in contiguous LIFO order by adjusting a single hardware stack-pointer register in `O(1)` time with strong L1 cache locality. Heap memory supports dynamically sized data structures whose lifetimes outlive the creating function call, trading allocator metadata overhead for runtime flexibility.",
    ),
    (
        "sorted input preconditions in binary search",
        "Binary search achieves `O(log N)` lookup time because comparing the target against the middle element eliminates half of the remaining array in a single check, which is only valid when global sorted order holds. On unsorted data, the midpoint comparison provides no guarantee about which half contains the target, so discarding either half can miss the element completely.",
    ),
    (
        "collision resolution in hash tables",
        "Because a hash function compresses a vast key space into a finite bucket array, distinct keys inevitably map to the same bucket index by the Pigeonhole Principle. Collision resolution schemes such as separate chaining or linear probing preserve all colliding key-value pairs accurately while maintaining `O(1)` average lookup performance.",
    ),
    (
        "mutexes and atomic operations on shared mutable state",
        "When multiple threads concurrently read, modify, and write a shared variable without synchronization, interleaved CPU instructions create race conditions that silently corrupt state. Mutexes and hardware atomic instructions serialize conflicting updates and enforce cache-coherent memory visibility across cores.",
    ),
    (
        "64-byte cache-line alignment and Struct-of-Arrays (`SoA`) memory layout",
        "Modern processors transfer memory in 64-byte cache lines and execute wide SIMD vector instructions across contiguous lanes. Storing each field in its own 64-byte-aligned contiguous array (`SoA`) ensures vector loads pack 100% useful elements without strided gather overhead or cross-cache-line splits.",
    ),
    (
        "arena (bump) allocators in low-latency systems",
        "General heap allocators maintain free lists and locks on every individual `malloc` and `free`, whereas an arena allocator reserves one contiguous block up front and allocates in `O(1)` time by advancing an integer offset. Resetting the offset frees thousands of frame- or request-scoped objects in a single instruction with zero fragmentation.",
    ),
    (
        "virtual memory and hardware page tables",
        "Virtual memory gives each process an isolated logical address space mapped to physical RAM frames by the CPU's Memory Management Unit (MMU) and Translation Lookaside Buffer (TLB). Hardware page permissions prevent stray pointers in one process from reading or corrupting another process or the operating system kernel.",
    ),
    (
        "amortized capacity doubling in dynamic arrays",
        "Growing a full array by a constant number of slots copies existing elements repeatedly and degrades `N` appends to `O(N^2)` work, whereas doubling capacity spaces reallocations exponentially far apart so total copies remain below `2N`. This preserves contiguous cache-friendly indexing alongside `O(1)` amortized append complexity.",
    ),
    (
        "consistent lock acquisition ordering for deadlock prevention",
        "Deadlock arises when threads form a wait cycle—such as Thread A holding Lock 1 while waiting on Lock 2 while Thread B holds Lock 2 and waits on Lock 1. Enforcing a global lock order across every code path breaks circular wait, guaranteeing that at least one thread can always finish and release its locks.",
    ),
    (
        "epsilon tolerances in floating-point comparisons",
        "IEEE-754 binary floating-point numbers have finite significand precision and cannot represent many decimal fractions like `0.1` exactly, causing tiny rounding differences across operations. Checking `|a - b| <= eps` tests numerical agreement within machine precision instead of failing spuriously on `a == b`.",
    ),
    (
        "short-circuit evaluation in boolean expressions",
        "Short-circuit operators (`&&`/`and` and `||`/`or`) skip evaluating their right-hand operand whenever the left-hand operand already determines the boolean result. This enables safe guard idioms such as `ptr != NULL && ptr->ready` where evaluating the second expression unconditionally would fault.",
    ),
    (
        "power-of-two ring buffer capacities",
        "Integer modulo (`%`) emits a multi-cycle hardware division instruction, whereas wrapping an index modulo a power-of-two capacity `N` compiles to a single-cycle bitwise mask `idx & (N - 1)`. High-throughput audio, network, and GPU command rings use power-of-two sizes to minimize per-packet latency.",
    ),
    (
        "visited-set tracking in graph BFS and DFS",
        "Graphs Unlike trees can contain cycles and shared diamond paths, so traversing neighbors without recording visited vertices causes search routines to loop endlessly around cycles. Marking vertices as visited upon discovery bounds work to `O(V + E)` time.",
    ),
    (
        "write-ahead logging (WAL) in storage engines",
        "Updating B-tree pages directly in place risks leaving partially written structures on disk if power fails mid-transaction. Appending sequential redo records to a durable write-ahead log first guarantees atomic crash recovery while turning random commits into fast sequential I/O.",
    ),
    (
        "BuildBit (`bbc`) `gpu_arena` and `soa struct` compilation",
        "BuildBit natively compiles `soa struct` definitions into 64-byte aligned separate field arrays inside a lock-free `gpu_arena` so `parallel for` loops map directly onto wide AVX-512 SIMD lanes or zero-copy AMDGPU compute wavefronts without per-object heap allocation or AoS shuffling.",
    ),
    (
        "chunkwise Compact WY parallelization in linear associative memory",
        "Sequential delta-rule fast-weight updates have step-by-step matrix dependencies, but partitioning sequences into chunks of size `C=64` and solving a small `64x64` unit lower-triangular Compact WY system converts intra-chunk updates into dense matrix multiplications while keeping cross-chunk state `O(1)` in sequence length.",
    ),
]


def generate_conceptual_prose_turn(rng: random.Random) -> Tuple[str, str]:
    topic, core_body = rng.choice(CS_CONCEPT_TOPICS)
    q_templates = [
        f"Explain why {topic} is important in computer science.",
        f"In clear plain English, explain the role of {topic}.",
        f"Why do software engineers rely on {topic}?",
        f"What problem does {topic} solve, and what happens without it?",
    ]
    clean_body = core_body.replace("Graphs Unlike trees", "Unlike trees, general graphs")
    return rng.choice(q_templates), clean_body


def generate_polyglot_systems_turn(rng: random.Random) -> Tuple[str, str]:
    lang_choice = rng.choice(["buildbit", "c99", "cpp20", "python", "lua"])
    if lang_choice == "buildbit":
        structs = [
            ("WavePacket", [("px", "f32"), ("py", "f32"), ("vx", "f32"), ("vy", "f32")], "px", "vx", "0.0125"),
            ("RigidBody", [("pos", "f32"), ("vel", "f32"), ("force", "f32"), ("inv_mass", "f32")], "vel", "force", "0.008"),
            ("TelemetryNode", [("temp", "f32"), ("drift", "f32"), ("gain", "f32")], "temp", "drift", "0.05"),
            ("AudioVoice", [("phase", "f32"), ("freq", "f32"), ("amp", "f32")], "phase", "freq", "0.0000208"),
            ("BoidAgent", [("x", "f32"), ("y", "f32"), ("dx", "f32"), ("dy", "f32")], "x", "dx", "0.016"),
        ]
        sname, fields, dst_f, src_f, dt = rng.choice(structs)
        count = rng.choice([2048, 8192, 16384, 32768, 65536])
        arena_mb = rng.choice([8, 32, 64])
        arena_bytes = arena_mb * 1048576
        f_decl = ", ".join(f"{fn}: {ft}" for fn, ft in fields)
        prompt = (
            f"Write a BuildBit (.bb) program that defines `soa struct {sname}` with fields `{f_decl}`, "
            f"allocates {count} elements in a {arena_mb} MB `gpu_arena`, and updates `{dst_f}` in a `parallel for` loop."
        )
        code = (
            f"```rust\n"
            f"soa struct {sname} {{\n"
            f"    {f_decl}\n"
            f"}}\n\n"
            f"fn step_{sname.lower()}() {{\n"
            f"    gpu_arena mem = {arena_bytes}\n"
            f"    let items = alloc_soa(mem, {sname}, {count})\n"
            f"    parallel for i in 0..{count} {{\n"
            f"        items[i].{dst_f} = items[i].{dst_f} + items[i].{src_f} * {dt}\n"
            f"    }}\n"
            f"    reset(mem)\n"
            f"}}\n"
            f"```"
        )
        return prompt, code

    if lang_choice == "c99":
        tasks = [
            (
                "Write a C99 function `clamp_f32_slice` that clamps every element of a `float` array to `[lo, hi]` in place.",
                "```c\n#include <stddef.h>\n\nvoid clamp_f32_slice(float * restrict data, size_t len, float lo, float hi) {\n    if (!data) return;\n    for (size_t i = 0; i < len; ++i) {\n        float v = data[i];\n        if (v < lo) v = lo;\n        else if (v > hi) v = hi;\n        data[i] = v;\n    }\n}\n```",
            ),
            (
                "Write a C99 function `fnv1a_hash64` that computes the 64-bit FNV-1a hash of a byte buffer.",
                "```c\n#include <stdint.h>\n#include <stddef.h>\n\nuint64_t fnv1a_hash64(const uint8_t *buf, size_t len) {\n    uint64_t hash = 14695981039346656037ULL;\n    for (size_t i = 0; i < len; ++i) {\n        hash ^= (uint64_t)buf[i];\n        hash *= 1099511628211ULL;\n    }\n    return hash;\n}\n```",
            ),
            (
                "Write a C99 function `rotate_left_u32_array` that rotates an array of `uint32_t` values left by one position in place.",
                "```c\n#include <stdint.h>\n#include <stddef.h>\n\nvoid rotate_left_u32_array(uint32_t *arr, size_t n) {\n    if (!arr || n <= 1) return;\n    uint32_t first = arr[0];\n    for (size_t i = 0; i + 1 < n; ++i) {\n        arr[i] = arr[i + 1];\n    }\n    arr[n - 1] = first;\n}\n```",
            ),
            (
                "Write a C99 function `dot_product_f64` that computes the dot product of two `double` arrays.",
                "```c\n#include <stddef.h>\n\ndouble dot_product_f64(const double * restrict a, const double * restrict b, size_t n) {\n    double acc = 0.0;\n    for (size_t i = 0; i < n; ++i) {\n        acc += a[i] * b[i];\n    }\n    return acc;\n}\n```",
            ),
            (
                "Write a C99 function `binary_search_i64` that returns the index of `target` in a sorted `int64_t` array, or `-1` if absent.",
                "```c\n#include <stdint.h>\n#include <stddef.h>\n\nptrdiff_t binary_search_i64(const int64_t *arr, size_t n, int64_t target) {\n    size_t lo = 0, hi = n;\n    while (lo < hi) {\n        size_t mid = lo + ((hi - lo) >> 1);\n        if (arr[mid] == target) return (ptrdiff_t)mid;\n        if (arr[mid] < target) lo = mid + 1;\n        else hi = mid;\n    }\n    return -1;\n}\n```",
            ),
        ]
        return rng.choice(tasks)

    if lang_choice == "cpp20":
        cap = rng.choice([64, 256, 512, 1024])
        cname = rng.choice(["RingQueue", "BoundedBuffer", "CircularFifo"])
        prompt = f"Write a C++20 generic class template `{cname}<T, {cap}>` implementing a fixed-capacity FIFO ring buffer with `push`, `pop`, and `size`."
        code = (
            f"```cpp\n#include <array>\n#include <cstddef>\n#include <stdexcept>\n\n"
            f"template <typename T, std::size_t Capacity = {cap}>\n"
            f"class {cname} {{\n"
            f"    std::array<T, Capacity> storage_{{}};\n"
            f"    std::size_t head_ = 0;\n"
            f"    std::size_t tail_ = 0;\n"
            f"    std::size_t count_ = 0;\n\n"
            f"public:\n"
            f"    bool push(const T& item) noexcept {{\n"
            f"        if (count_ == Capacity) return false;\n"
            f"        storage_[tail_] = item;\n"
            f"        tail_ = (tail_ + 1) % Capacity;\n"
            f"        ++count_;\n"
            f"        return true;\n"
            f"    }}\n\n"
            f"    T pop() {{\n"
            f"        if (count_ == 0) throw std::underflow_error(\"{cname} is empty\");\n"
            f"        T val = storage_[head_];\n"
            f"        head_ = (head_ + 1) % Capacity;\n"
            f"        --count_;\n"
            f"        return val;\n"
            f"    }}\n\n"
            f"    [[nodiscard]] constexpr std::size_t size() const noexcept {{ return count_; }}\n"
            f"}};\n```"
        )
        return prompt, code

    if lang_choice == "lua":
        cls_specs = [
            ("Vec2", "x, y", "self.x = x or 0.0\n    self.y = y or 0.0", "scale(factor)", "self.x = self.x * factor\n    self.y = self.y * factor"),
            ("SpringDamper", "pos, stiffness", "self.pos = pos or 0.0\n    self.vel = 0.0\n    self.stiffness = stiffness or 12.0", "step(target, dt)", "local force = (target - self.pos) * self.stiffness\n    self.vel = (self.vel + force * dt) * 0.92\n    self.pos = self.pos + self.vel * dt"),
            ("RateLimiter", "capacity, rate", "self.capacity = capacity or 10.0\n    self.tokens = self.capacity\n    self.rate = rate or 2.0", "refill(dt)", "self.tokens = math.min(self.capacity, self.tokens + self.rate * dt)"),
            ("CounterMetric", "name", "self.name = name or 'metric'\n    self.count = 0", "add(delta)", "self.count = self.count + (delta or 1)"),
        ]
        cname, args, init_body, m_sig, m_body = rng.choice(cls_specs)
        prompt = f"Write a Lua module implementing a `{cname}` class using metatables (`__index`, `.new({args})`, and `:{m_sig}`)."
        code = (
            f"```lua\nlocal {cname} = {{}}\n{cname}.__index = {cname}\n\n"
            f"function {cname}.new({args})\n"
            f"    local self = setmetatable({{}}, {cname})\n"
            f"    {init_body}\n"
            f"    return self\n"
            f"end\n\n"
            f"function {cname}:{m_sig}\n"
            f"    {m_body}\n"
            f"    return self\n"
            f"end\n\n"
            f"return {cname}\n```"
        )
        return prompt, code

    # Python clean algorithmic implementation
    py_tasks = [
        (
            "[python] Write a Python function `merge_intervals` that merges all overlapping `[start, end]` intervals.",
            "```python\ndef merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    if not intervals:\n        return []\n    ordered = sorted(intervals, key=lambda pair: pair[0])\n    merged = [ordered[0][:]]\n    for start, end in ordered[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n```",
        ),
        (
            "[python] Write a Python function `running_mean_and_min` that returns the minimum element and arithmetic mean of a non-empty list of floats.",
            "```python\ndef running_mean_and_min(values: list[float]) -> tuple[float, float]:\n    if not values:\n        raise ValueError('values must not be empty')\n    smallest = values[0]\n    total = 0.0\n    for v in values:\n        if v < smallest:\n            smallest = v\n        total += v\n    return smallest, total / len(values)\n```",
        ),
        (
            "[python] Write a Python function `topological_sort_kahn` that returns a topological ordering of a directed acyclic graph.",
            "```python\nfrom collections import deque\n\ndef topological_sort_kahn(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:\n    adj = [[] for _ in range(num_nodes)]\n    indeg = [0] * num_nodes\n    for u, v in edges:\n        adj[u].append(v)\n        indeg[v] += 1\n    q = deque(i for i in range(num_nodes) if indeg[i] == 0)\n    order = []\n    while q:\n        u = q.popleft()\n        order.append(u)\n        for v in adj[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    if len(order) != num_nodes:\n        raise ValueError('Graph contains a cycle')\n    return order\n```",
        ),
    ]
    return rng.choice(py_tasks)


def generate_rate_math_turn(rng: random.Random) -> Tuple[str, str, str]:
    """Returns `(prompt, think_steps, final_answer_line)` strictly holding out 45 * 4."""
    kind = rng.choice(["kinematics", "throughput", "energy", "warehouse"])
    if kind == "kinematics":
        veh = rng.choice(["probe", "freight train", "autonomous drone", "courier van", "research vessel"])
        speed = rng.choice([28, 32, 36, 42, 48, 55, 62, 65, 72, 85, 110])
        hours = rng.choice([3, 5, 6, 7, 8, 9, 12])
        dist = speed * hours
        prompt = f"If a {veh} travels at a constant speed of {speed} miles per hour for {hours} hours, how far does it travel in total?"
        think = (
            f"Distance equals speed multiplied by time.\n"
            f"Speed = {speed} miles/hour\n"
            f"Time = {hours} hours\n"
            f"Total distance = {speed} * {hours} = {dist} miles."
        )
        ans = f"The {veh} travels {dist} miles in total."
        return prompt, think, ans

    if kind == "throughput":
        rate = rng.choice([120, 250, 340, 480, 640, 750])
        secs = rng.choice([12, 15, 18, 24, 30, 45])
        total = rate * secs
        prompt = f"A telemetry bus processes {rate} packets per second. How many packets does it process in {secs} seconds?"
        think = (
            f"Multiply the packet rate by the elapsed duration.\n"
            f"Rate = {rate} packets/second\n"
            f"Duration = {secs} seconds\n"
            f"Total packets = {rate} * {secs} = {total}."
        )
        ans = f"It processes {total} packets in {secs} seconds."
        return prompt, think, ans

    if kind == "energy":
        watts = rng.choice([65, 85, 120, 150, 220, 350])
        hrs = rng.choice([4, 6, 8, 10, 15, 24])
        wh = watts * hrs
        prompt = f"A compute blade draws {watts} watts continuously for {hrs} hours. How many watt-hours of energy does it consume?"
        think = (
            f"Energy in watt-hours equals power in watts multiplied by time in hours.\n"
            f"Power = {watts} W, Time = {hrs} h\n"
            f"Energy = {watts} * {hrs} = {wh} watt-hours."
        )
        ans = f"The compute blade consumes {wh} watt-hours of energy."
        return prompt, think, ans

    boxes = rng.choice([14, 18, 22, 26, 34])
    per_box = rng.choice([12, 16, 24, 30])
    extra = rng.choice([5, 9, 13, 17])
    tot = boxes * per_box + extra
    prompt = f"An engineer packs {boxes} crates with {per_box} sensors in each crate, plus {extra} spare sensors in a carry case. How many sensors are there altogether?"
    think = (
        f"Sensors in crates = {boxes} * {per_box} = {boxes * per_box}.\n"
        f"Add the {extra} spare sensors: {boxes * per_box} + {extra} = {tot}."
    )
    ans = f"There are {tot} sensors altogether."
    return prompt, think, ans


# ==============================================================================
# 3. DIVERSE MULTI-SPAN 2K CROSS-CHUNK ASSOCIATIVE RETRIEVAL SYNTHESIZER
#    Spans 1,300 .. 1,980 tokens (`20 .. 31` chunks) with 1..35 token key-value gaps!
# ==============================================================================

TRAIN_ENTITY_STEMS = [
    "AeroValve", "FluxCoil", "CryoPump", "NavMatrix", "PulseCore", "OpticBus",
    "GyroArray", "PhaseLink", "MagLatch", "VoltRail", "LaserSync", "ChronoClk",
    "HydroSeal", "QuantumMux", "Piezoelec", "SpectraCam", "ServoAxis", "TachyonRx",
    "VortexFan", "OmniRelay", "VectorPLL", "TensorDMA", "PhotonBus", "WaveGuide",
    "CacheRing", "FuseBlock", "BusArbiter", "ClockTree", "PowerGate", "SyncFIFO",
    "DeltaDAC", "SigmaADC", "MicroPLL", "HyperLink", "CoreWatch", "FrameSync",
]
assert HOLDOUT_NIAH_PREFIXES.isdisjoint(set(TRAIN_ENTITY_STEMS)), "Entity leak!"

FILLER_SENTENCES = [
    "All redundant bus lanes completed cyclic redundancy verification with zero bit-flip events across the sampling window.",
    "Secondary harmonic dampers remained well within nominal mechanical tolerance during high-slew attitude maneuvers.",
    "Housekeeping telemetry packets were buffered in ring memory and flushed cleanly over the optical downlink channel.",
    "Ambient bus voltage ripple measured below 1.4 millivolts peak-to-peak under full multi-core vector load.",
    "Watchdog heartbeat counters incremented monotonically across every supervisory microcontroller on the backplane.",
    "Thermal heat-pipe conductance stayed balanced across radiators Alpha through Delta with zero throttling.",
    "ECC scrubbing engines scanned 64 megabytes of SRAM without encountering any uncorrectable multi-bit syndrome.",
    "Phase-locked oscillator drift remained below 0.03 parts per million relative to the rubidium reference clock.",
]


def synthesize_2k_multi_span_niah(rng: random.Random) -> Tuple[str, List[Tuple[str, str]], str]:
    """
    Generates a 1,350..1,920 token multi-chunk retrieval task across 3 diverse structural schemas:
      - Schema A: Subsystem Telemetry Manifests with 12..25 token identical-boilerplate gaps!
      - Schema B: Multi-File `<file_sep>` Codebase Constant / Config Lookups!
      - Schema C: Prose Paragraphs + Structured Engineering Registry Tables (25..40 token gaps)!
    Returns `(user_prompt, [(key_name, exact_val), ...], assistant_prefix_and_lines)`.
    """
    schema = rng.choice(["telemetry_manifest", "telemetry_manifest", "multifile_code", "prose_spec"])
    n_sections = rng.randint(15, 21)
    stems = rng.sample(TRAIN_ENTITY_STEMS, k=min(n_sections, len(TRAIN_ENTITY_STEMS)))
    code_prefix = rng.choice(["CHK", "VRF", "TOK", "KEY", "SIG", "HEX"])

    items: List[Tuple[str, str]] = []
    used_vals = set()
    for idx, st in enumerate(stems):
        suffix = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 9)}"
        ent = f"{st}-{suffix}"
        while True:
            if code_prefix == "HEX":
                val = f"0x{rng.randint(0x1000, 0xFFFF):04X}"
            else:
                val = f"{code_prefix}-{rng.randint(1000, 9999)}"
            if val not in used_vals:
                used_vals.add(val)
                break
        items.append((ent, val))

    n_queries = rng.choice([2, 2, 2, 3])
    q_indices = sorted(rng.sample(range(len(items)), k=n_queries))
    queried = [items[i] for i in q_indices]

    if schema == "telemetry_manifest":
        blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
        gap_style = rng.choice(["boilerplate_15tok", "wide_28tok", "compact_4tok"])
        for idx, (ent, val) in enumerate(items, start=1):
            f1, f2 = rng.sample(FILLER_SENTENCES, k=2)
            status = rng.choice(["NOMINAL", "OPTIMAL", "VERIFIED", "ACTIVE"])
            if gap_style == "boilerplate_15tok":
                header = f"[Subsystem {idx:02d}: {ent}] Status: {status} | Verification Code: {val}"
            elif gap_style == "wide_28tok":
                temp_k = rng.randint(285, 330)
                lane = rng.randint(0, 7)
                header = (
                    f"[Subsystem {idx:02d}: {ent}] Operational State: {status} | Thermal Sensor: {temp_k}K "
                    f"| PCIe Lane Group: {lane} | Assigned Verification Code: {val}"
                )
            else:
                header = f"[Subsystem {idx:02d}] {ent} -> {val} ({status})"
            blocks.append(f"{header}\n  Log: {f1} {f2}")

        q_names = " and ".join(ent for ent, _ in queried) if len(queried) == 2 else (
            ", ".join(ent for ent, _ in queried[:-1]) + f", and {queried[-1][0]}"
        )
        question = (
            f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
            f"for {q_names}? List each subsystem and its exact code."
        )
        return "\n".join(blocks) + question, queried, "telemetry"

    if schema == "multifile_code":
        files = []
        for idx, (ent, val) in enumerate(items, start=1):
            ident = ent.replace("-", "_").upper()
            f1 = rng.choice(FILLER_SENTENCES)
            files.append(
                f"<file_sep>src/mod_{idx:02d}_{ent.lower().replace('-', '_')}.h\n"
                f"// Module: {ent} — {f1}\n"
                f"#pragma once\n"
                f"#define {ident}_CHANNEL_ID {idx}\n"
                f"#define {ident}_GUARD_TOKEN \"{val}\"\n"
                f"static inline int init_{ent.lower().replace('-', '_')}(void) {{ return {idx}; }}\n"
            )
        q_names = " and ".join(ent for ent, _ in queried) if len(queried) == 2 else (
            ", ".join(ent for ent, _ in queried[:-1]) + f", and {queried[-1][0]}"
        )
        question = (
            f"\n\nQuestion: Across the repository header files above, what are the exact `GUARD_TOKEN` string values "
            f"defined for modules {q_names}?"
        )
        return "\n".join(files) + question, queried, "codebase"

    # Prose specification paragraphs with ~25-35 token natural-language span between entity and value
    paras = ["ENGINEERING QUALIFICATION REPORT\n"]
    for idx, (ent, val) in enumerate(items, start=1):
        f1 = rng.choice(FILLER_SENTENCES)
        paras.append(
            f"Section {idx:02d}: During environmental qualification testing, component {ent} completed full thermal-vacuum "
            f"cycling across twenty-four orbital periods without bus contention and was assigned calibration signature {val}. {f1}"
        )
    q_names = " and ".join(ent for ent, _ in queried) if len(queried) == 2 else (
        ", ".join(ent for ent, _ in queried[:-1]) + f", and {queried[-1][0]}"
    )
    question = (
        f"\n\nQuestion: According to the qualification report above, what are the exact calibration signatures "
        f"assigned to {q_names}?"
    )
    return "\n\n".join(paras) + question, queried, "prose_spec"


# ==============================================================================
# 4. CHATML SPAN ENCODER WITH EXACT PER-TOKEN LOSS MASKS
# ==============================================================================

class VaultV2TokenBuilder:
    def __init__(self, tok):
        self.tok = tok
        self.eos_id = tok.convert_tokens_to_ids("<|endoftext|>")
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        assert 0 <= self.eos_id < 65535 and 0 <= self.im_start_id < 65535 and 0 <= self.im_end_id < 65535

        self.user_prefix = tok.encode("user\n", add_special_tokens=False)
        self.assist_prefix = tok.encode("\n", add_special_tokens=False)
        self.assist_header = [self.im_start_id] + tok.encode("assistant\n", add_special_tokens=False)

    def encode_chatml_turn(
        self,
        prompt: str,
        response: str,
        sft_mode: bool = False,
        high_salience_suffix: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns `(token_ids, loss_weights_for_next_token)` of identical length `N`,
        where `w[i]` is the weight applied when predicting `token_ids[i]` from `token_ids[:i]`.
        """
        p_ids = self.tok.encode(prompt.strip(), add_special_tokens=False)
        u_block = [self.im_start_id] + self.user_prefix + p_ids + [self.im_end_id] + self.assist_prefix + self.assist_header
        u_w = [W_IGNORE if sft_mode else W_PROMPT] * len(u_block)
        u_w[0] = W_IGNORE

        if high_salience_suffix and response.endswith(high_salience_suffix):
            prefix_part = response[: len(response) - len(high_salience_suffix)]
            r_pre_ids = self.tok.encode(prefix_part, add_special_tokens=False)
            r_suf_ids = self.tok.encode(high_salience_suffix, add_special_tokens=False)
            a_ids = r_pre_ids + r_suf_ids + [self.im_end_id, self.eos_id]
            a_w = ([W_ASSIST] * len(r_pre_ids)) + ([W_NEEDLE] * len(r_suf_ids)) + [W_NEEDLE, W_IGNORE]
        else:
            r_ids = self.tok.encode(response.strip(), add_special_tokens=False)
            a_ids = r_ids + [self.im_end_id, self.eos_id]
            a_w = ([W_ASSIST] * len(r_ids)) + [W_NEEDLE, W_IGNORE]

        return u_block + a_ids, u_w + a_w

    def encode_niah_turn(
        self,
        prompt: str,
        queried: List[Tuple[str, str]],
        sft_mode: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """
        Weights each retrieved value (`CHK-xxxx`) and `<|im_end|>` at `W_NEEDLE = 30` (`3.0x`)!
        """
        p_ids = self.tok.encode(prompt.strip(), add_special_tokens=False)
        u_block = [self.im_start_id] + self.user_prefix + p_ids + [self.im_end_id] + self.assist_prefix + self.assist_header
        u_w = [W_IGNORE if sft_mode else W_PROMPT] * len(u_block)
        u_w[0] = W_IGNORE

        a_ids: List[int] = []
        a_w: List[int] = []
        for idx, (ent, val) in enumerate(queried):
            lead = ("" if idx == 0 else "\n") + f"- {ent}: "
            lead_ids = self.tok.encode(lead, add_special_tokens=False)
            val_ids = self.tok.encode(val, add_special_tokens=False)
            a_ids.extend(lead_ids + val_ids)
            a_w.extend(([W_ASSIST] * len(lead_ids)) + ([W_NEEDLE] * len(val_ids)))

        a_ids.extend([self.im_end_id, self.eos_id])
        a_w.extend([W_NEEDLE, W_IGNORE])
        return u_block + a_ids, u_w + a_w


# ==============================================================================
# 5. DOCUMENT-AWARE ROW PACKER (NEVER SPLITS QA / NIAH TURNS ACROSS ROWS!)
# ==============================================================================

class SequenceRowPacker:
    """
    Packs `(ids, w)` items into exact `ROW_LEN = 2049` sequences (`x = ids[:2048], y = ids[1:2049], mask = w[1:2049]`).
    Atomic ChatML / NIAH turns <= 2049 tokens NEVER split across two rows!
    """
    def __init__(self, eos_id: int):
        self.eos_id = eos_id
        self.rows_tokens: List[np.ndarray] = []
        self.rows_masks: List[np.ndarray] = []
        self.cur_ids: List[int] = []
        self.cur_w: List[int] = []

    def _flush_current_padded(self):
        if len(self.cur_ids) < 64:
            self.cur_ids.clear()
            self.cur_w.clear()
            return
        pad_n = ROW_LEN - len(self.cur_ids)
        if pad_n > 0:
            self.cur_ids.extend([self.eos_id] * pad_n)
            self.cur_w.extend([W_IGNORE] * pad_n)
        arr_t = np.asarray(self.cur_ids[:ROW_LEN], dtype=np.uint16)
        arr_m = np.asarray(self.cur_w[1:ROW_LEN], dtype=np.uint8)
        if int(arr_m.sum()) > 0:
            self.rows_tokens.append(arr_t)
            self.rows_masks.append(arr_m)
        self.cur_ids.clear()
        self.cur_w.clear()

    def add_atomic_turn(self, ids: List[int], w: List[int]):
        if len(ids) > ROW_LEN:
            return  # Skip overlong QA turns rather than truncating the answer!
        if len(self.cur_ids) + len(ids) > ROW_LEN:
            self._flush_current_padded()
        self.cur_ids.extend(ids)
        self.cur_w.extend(w)
        if len(self.cur_ids) >= ROW_LEN - 32:
            self._flush_current_padded()

    def add_continuous_doc(self, ids: List[int], w_val: int = W_PROSE):
        """Packs long textbooks / web articles / raw code files seamlessly across 2049-token chunks."""
        pos = 0
        n = len(ids)
        while pos < n:
            space = ROW_LEN - len(self.cur_ids)
            take = min(space, n - pos)
            chunk = ids[pos : pos + take]
            w_chunk = [w_val] * take
            if pos == 0 and take > 0:
                w_chunk[0] = W_IGNORE
            self.cur_ids.extend(chunk)
            self.cur_w.extend(w_chunk)
            pos += take
            if len(self.cur_ids) == ROW_LEN:
                arr_t = np.asarray(self.cur_ids, dtype=np.uint16)
                arr_m = np.asarray(self.cur_w[1:], dtype=np.uint8)
                self.rows_tokens.append(arr_t)
                self.rows_masks.append(arr_m)
                self.cur_ids.clear()
                self.cur_w.clear()
        if self.cur_ids and self.cur_ids[-1] != self.eos_id and len(self.cur_ids) < ROW_LEN:
            self.cur_ids.append(self.eos_id)
            self.cur_w.append(W_IGNORE)
            if len(self.cur_ids) == ROW_LEN:
                arr_t = np.asarray(self.cur_ids, dtype=np.uint16)
                arr_m = np.asarray(self.cur_w[1:], dtype=np.uint8)
                self.rows_tokens.append(arr_t)
                self.rows_masks.append(arr_m)
                self.cur_ids.clear()
                self.cur_w.clear()

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        self._flush_current_padded()
        if not self.rows_tokens:
            return np.zeros((0, ROW_LEN), dtype=np.uint16), np.zeros((0, SEQ_LEN), dtype=np.uint8)
        return np.stack(self.rows_tokens, axis=0), np.stack(self.rows_masks, axis=0)


# ==============================================================================
# 6. MAIN PROCUREMENT & PRE-TOKENIZATION PIPELINE
# ==============================================================================

def parse_metamath_cot(resp: str) -> Tuple[str, str]:
    if "The answer is:" in resp:
        parts = resp.rsplit("The answer is:", 1)
        steps = parts[0].strip()
        if "\n####" in steps:
            steps = steps.rsplit("\n####", 1)[0].strip()
        ans = parts[1].strip()
        return steps, f"Final Answer: {ans}"
    if "####" in resp:
        parts = resp.rsplit("####", 1)
        return parts[0].strip(), f"Final Answer: {parts[1].strip()}"
    lines = [ln.strip() for ln in resp.strip().splitlines() if ln.strip()]
    if len(lines) >= 2:
        return "\n".join(lines[:-1]), lines[-1]
    return resp.strip(), resp.strip()


def save_split_memmaps(
    out_dir: str,
    prefix: str,
    tokens_arr: np.ndarray,
    masks_arr: np.ndarray,
):
    tok_path = os.path.join(out_dir, f"{prefix}_tokens.bin")
    msk_path = os.path.join(out_dir, f"{prefix}_mask.bin")
    fp_t = np.memmap(tok_path, dtype=np.uint16, mode="w+", shape=tokens_arr.shape)
    fp_t[:] = tokens_arr[:]
    fp_t.flush()
    fp_m = np.memmap(msk_path, dtype=np.uint8, mode="w+", shape=masks_arr.shape)
    fp_m[:] = masks_arr[:]
    fp_m.flush()
    print(
        f"  [+] Wrote {prefix}: {tokens_arr.shape[0]:,} rows x {SEQ_LEN} ({tokens_arr.shape[0] * SEQ_LEN / 1e6:.2f}M tokens) "
        f"| {os.path.getsize(tok_path)/1e6:.1f} MB + {os.path.getsize(msk_path)/1e6:.1f} MB",
        flush=True,
    )


def main():
    t_start = time.perf_counter()
    rng = random.Random(20260202)
    np_rng = np.random.default_rng(20260202)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data_v2", "raw")
    out_dir = os.path.join(base_dir, "data_v2")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    builder = VaultV2TokenBuilder(tok)

    audit = {
        "rejected_code_review_poison": 0,
        "rejected_benchmark_holdout": 0,
        "rejected_too_short_or_overlong": 0,
        "domain_rows_pretrain": {},
        "domain_tokens_pretrain": {},
        "domain_rows_sft": {},
        "domain_tokens_sft": {},
    }

    print("=" * 96, flush=True)
    print("VAULT v2 DATA PROCUREMENT, DECONTAMINATION & NVME MEMMAP BUILDER (L = 2048)", flush=True)
    print("=" * 96, flush=True)

    # --------------------------------------------------------------------------
    # STREAM 1: EDUCATIONAL TEXTBOOKS & WEB PROSE (~92M tokens = ~45,000 rows)
    # --------------------------------------------------------------------------
    print("[1/5] Tokenizing Cosmopedia-v2 Textbooks + FineWeb-Edu-Dedup Prose...", flush=True)
    prose_packer = SequenceRowPacker(builder.eos_id)

    cosmo_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "cosmopedia-v2", "train-00000-of-00104.parquet")
    pf_cosmo = pq.ParquetFile(cosmo_path)
    cosmo_target_rows = 26500  # ~54.3M textbook tokens
    for rg_idx in range(pf_cosmo.num_row_groups):
        if len(prose_packer.rows_tokens) >= cosmo_target_rows:
            break
        batch_texts = pf_cosmo.read_row_group(rg_idx, columns=["text"])["text"].to_pylist()
        clean_texts = []
        for tx in batch_texts:
            if not tx or len(tx) < 200:
                continue
            tl_low = tx.lower()
            if any(h in tl_low for h in HOLDOUT_SUBSTRINGS[:6]):
                audit["rejected_benchmark_holdout"] += 1
                continue
            clean_texts.append(tx)
        for i in range(0, len(clean_texts), 1024):
            if len(prose_packer.rows_tokens) >= cosmo_target_rows:
                break
            enc = tok(clean_texts[i : i + 1024], add_special_tokens=False)["input_ids"]
            for ids in enc:
                if len(ids) >= 64:
                    prose_packer.add_continuous_doc(ids, w_val=W_PROSE)

    cosmo_built = len(prose_packer.rows_tokens)
    print(f"      -> Cosmopedia-v2 Textbooks: {cosmo_built:,} rows ({cosmo_built * SEQ_LEN / 1e6:.2f}M tokens)", flush=True)

    fw_path = os.path.join(raw_dir, "HuggingFaceTB__smollm-corpus", "fineweb-edu-dedup", "train-00000-of-00234.parquet")
    pf_fw = pq.ParquetFile(fw_path)
    prose_total_target = cosmo_built + 18500  # + ~37.9M FineWeb-Edu tokens = ~92.2M total textbook/edu tokens
    for rg_idx in range(pf_fw.num_row_groups):
        if len(prose_packer.rows_tokens) >= prose_total_target:
            break
        batch_texts = pf_fw.read_row_group(rg_idx, columns=["text"])["text"].to_pylist()
        clean_texts = [tx for tx in batch_texts if tx and len(tx) >= 250]
        for i in range(0, len(clean_texts), 1024):
            if len(prose_packer.rows_tokens) >= prose_total_target:
                break
            enc = tok(clean_texts[i : i + 1024], add_special_tokens=False)["input_ids"]
            for ids in enc:
                if len(ids) >= 80:
                    prose_packer.add_continuous_doc(ids, w_val=W_PROSE)

    prose_toks, prose_msks = prose_packer.finalize()
    print(f"      -> Total Stream 1 (Textbooks + FineWeb-Edu): {len(prose_toks):,} rows ({len(prose_toks) * SEQ_LEN / 1e6:.2f}M tokens)", flush=True)

    # --------------------------------------------------------------------------
    # STREAM 2: PURE CONVERSATIONAL & CONCEPTUAL PLAIN-ENGLISH Q&A (NO <think>!)
    # --------------------------------------------------------------------------
    print("[2/5] Building Pure Plain-English Conversational & Conceptual Q&A (SmolTalk + Science + CS Concepts)...", flush=True)
    conv_pre_packer = SequenceRowPacker(builder.eos_id)
    conv_sft_packer = SequenceRowPacker(builder.eos_id)

    allowed_cats = {"information-seeking", "reasoning", "advice-seeking", "planning", "brainstorming", "data-analysis"}
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    smoltalk_arrows = sorted(glob.glob(os.path.join(hf_cache, "datasets/HuggingFaceTB___smoltalk/**/*train-*.arrow"), recursive=True))
    conv_pairs: List[Tuple[str, str]] = []

    for ap in smoltalk_arrows:
        if len(conv_pairs) >= 65000:
            break
        with ipc.open_stream(ap) as reader:
            tbl = reader.read_all()
            msgs_col = tbl["messages"].to_pylist()
            cat_col = tbl["category"].to_pylist()
            qual_col = tbl["quality"].to_pylist()
            for msgs, cat, qual in zip(msgs_col, cat_col, qual_col):
                if qual not in ("excellent", "good") or cat not in allowed_cats:
                    continue
                if not msgs or len(msgs) < 2:
                    continue
                u_msg = msgs[0].get("content", "") if msgs[0].get("role") == "user" else ""
                a_msg = msgs[1].get("content", "") if msgs[1].get("role") == "assistant" else ""
                if "```" in a_msg or "<think>" in a_msg:
                    continue
                ok, reason = is_clean_pair(u_msg, a_msg)
                if not ok:
                    if reason == "code_review_poison":
                        audit["rejected_code_review_poison"] += 1
                    elif reason == "benchmark_holdout":
                        audit["rejected_benchmark_holdout"] += 1
                    continue
                conv_pairs.append((u_msg, a_msg))
                if len(conv_pairs) >= 65000:
                    break

    # Add crisp CS & Systems Plain-English Conceptual Q&A (strictly zero <think>, zero code fences, zero recursion holdout!)
    for _ in range(3200):
        conv_pairs.append(generate_conceptual_prose_turn(rng))

    rng.shuffle(conv_pairs)
    for idx, (u_txt, a_txt) in enumerate(conv_pairs):
        if len(conv_pre_packer.rows_tokens) >= 22000 and len(conv_sft_packer.rows_tokens) >= 4200:
            break
        if len(conv_pre_packer.rows_tokens) < 22000:
            ids_p, w_p = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=False)
            conv_pre_packer.add_atomic_turn(ids_p, w_p)
        if idx % 4 == 0 and len(conv_sft_packer.rows_tokens) < 4200:
            ids_s, w_s = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            conv_sft_packer.add_atomic_turn(ids_s, w_s)

    conv_pre_toks, conv_pre_msks = conv_pre_packer.finalize()
    conv_sft_toks, conv_sft_msks = conv_sft_packer.finalize()
    print(
        f"      -> Stream 2 Plain-English ChatML: Pretrain={len(conv_pre_toks):,} rows ({len(conv_pre_toks)*SEQ_LEN/1e6:.2f}M tok) "
        f"| SFT={len(conv_sft_toks):,} rows ({len(conv_sft_toks)*SEQ_LEN/1e6:.2f}M tok)",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # STREAM 3: POLYGLOT FROM-SCRATCH CODE + BUILDBIT (.bb) REPOSITORIES
    # --------------------------------------------------------------------------
    print("[3/5] Building Decontaminated Polyglot Code (Magicoder + CodeFeedback + EvolCode + BuildBit .bb)...", flush=True)
    code_pre_packer = SequenceRowPacker(builder.eos_id)
    code_sft_packer = SequenceRowPacker(builder.eos_id)

    # 3A. Authentic Philip Bartsch BuildBit (.bb) source files & docs
    bb_files = sorted(
        set(
            glob.glob(os.path.join(base_dir, "..", "buildbit", "**", "*.bb"), recursive=True)
            + glob.glob(os.path.join(base_dir, "..", "buildbit-os", "**", "*.bb"), recursive=True)
            + [
                os.path.join(base_dir, "..", "buildbit", "README.md"),
                os.path.join(base_dir, "..", "buildbit-os", "README.md"),
            ]
        )
    )
    bb_raw_docs = []
    for bf in bb_files:
        if os.path.isfile(bf):
            with open(bf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if len(content) >= 80 and "soa struct photon" not in content.lower():
                rel = os.path.basename(bf)
                bb_raw_docs.append(f"<file_sep>{rel}\n{content}")

    for _ in range(12):  # Gentle multi-offset coverage across real BuildBit OS & compiler kernels
        rng.shuffle(bb_raw_docs)
        for doc in bb_raw_docs:
            ids = tok.encode(doc, add_special_tokens=False)
            code_pre_packer.add_continuous_doc(ids, w_val=W_PROSE)

    # 3B. Decontaminated public from-scratch coding pairs + procedural systems synthesis
    code_pairs: List[Tuple[str, str]] = []

    magi_path = os.path.join(raw_dir, "ise-uiuc__Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl")
    with open(magi_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ok, reason = is_clean_pair(r.get("problem", ""), r.get("solution", ""))
            if not ok:
                if reason == "code_review_poison":
                    audit["rejected_code_review_poison"] += 1
                elif reason == "benchmark_holdout":
                    audit["rejected_benchmark_holdout"] += 1
                continue
            code_pairs.append((r["problem"], r["solution"]))

    cf_path = os.path.join(raw_dir, "m-a-p__CodeFeedback-Filtered-Instruction", "CodeFeedback-Filtered-Instruction.jsonl")
    with open(cf_path, "r", encoding="utf-8") as f:
        for idx_l, line in enumerate(f):
            if idx_l >= 70000:
                break
            r = json.loads(line)
            ok, reason = is_clean_pair(r.get("query", ""), r.get("answer", ""))
            if not ok:
                if reason == "code_review_poison":
                    audit["rejected_code_review_poison"] += 1
                elif reason == "benchmark_holdout":
                    audit["rejected_benchmark_holdout"] += 1
                continue
            code_pairs.append((r["query"], r["answer"]))

    evol_path = os.path.join(raw_dir, "nickrosh__Evol-Instruct-Code-80k-v1", "EvolInstruct-Code-80k.json")
    with open(evol_path, "r", encoding="utf-8") as f:
        evol_data = json.load(f)
    for r in evol_data[:35000]:
        ok, reason = is_clean_pair(r.get("instruction", ""), r.get("output", ""))
        if not ok:
            if reason == "code_review_poison":
                audit["rejected_code_review_poison"] += 1
            elif reason == "benchmark_holdout":
                audit["rejected_benchmark_holdout"] += 1
            continue
        code_pairs.append((r["instruction"], r["output"]))

    # Diverse parameterized BuildBit (.bb), C99, C++20, Lua, Python systems modules
    for _ in range(9000):
        code_pairs.append(generate_polyglot_systems_turn(rng))

    rng.shuffle(code_pairs)
    for idx, (u_txt, a_txt) in enumerate(code_pairs):
        if len(code_pre_packer.rows_tokens) >= 29500 and len(code_sft_packer.rows_tokens) >= 5400:
            break
        if len(code_pre_packer.rows_tokens) < 29500:
            ids_p, w_p = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=False)
            code_pre_packer.add_atomic_turn(ids_p, w_p)
        if idx % 4 == 0 and len(code_sft_packer.rows_tokens) < 5400:
            ids_s, w_s = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            code_sft_packer.add_atomic_turn(ids_s, w_s)

    code_pre_toks, code_pre_msks = code_pre_packer.finalize()
    code_sft_toks, code_sft_msks = code_sft_packer.finalize()
    print(
        f"      -> Stream 3 Polyglot Code: Pretrain={len(code_pre_toks):,} rows ({len(code_pre_toks)*SEQ_LEN/1e6:.2f}M tok) "
        f"| SFT={len(code_sft_toks):,} rows ({len(code_sft_toks)*SEQ_LEN/1e6:.2f}M tok)",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # STREAM 4: EXPLICIT <think>...</think> MATHEMATICAL & SCIENTIFIC REASONING
    # --------------------------------------------------------------------------
    print("[4/5] Building <think>...</think> Math & Quantitative Reasoning (MetaMathQA + MathInstruct + Rate Physics)...", flush=True)
    math_pre_packer = SequenceRowPacker(builder.eos_id)
    math_sft_packer = SequenceRowPacker(builder.eos_id)

    math_items: List[Tuple[str, str, str]] = []  # (prompt, full_response, final_suffix)

    mm_path = os.path.join(raw_dir, "meta-math__MetaMathQA", "MetaMathQA-395K.json")
    with open(mm_path, "r", encoding="utf-8") as f:
        mm_data = json.load(f)
    rng.shuffle(mm_data)
    for r in mm_data[:95000]:
        q = r.get("query", "")
        resp = r.get("response", "")
        ok, reason = is_clean_pair(q, resp)
        if not ok:
            if reason == "benchmark_holdout":
                audit["rejected_benchmark_holdout"] += 1
            continue
        steps, final_line = parse_metamath_cot(resp)
        formatted = f"<think>\n{steps}\n</think>\n{final_line}"
        math_items.append((q, formatted, final_line))

    mi_path = os.path.join(raw_dir, "TIGER-Lab__MathInstruct", "MathInstruct.json")
    with open(mi_path, "r", encoding="utf-8") as f:
        mi_data = json.load(f)
    rng.shuffle(mi_data)
    for r in mi_data[:45000]:
        q = r.get("instruction", "")
        out = r.get("output", "")
        if "```" in out:  # Keep PoT Python separate from pure <think> math derivations
            continue
        ok, reason = is_clean_pair(q, out)
        if not ok:
            continue
        steps, final_line = parse_metamath_cot(out)
        formatted = f"<think>\n{steps}\n</think>\n{final_line}"
        math_items.append((q, formatted, final_line))

    for _ in range(6000):
        q, steps, final_line = generate_rate_math_turn(rng)
        formatted = f"<think>\n{steps}\n</think>\n{final_line}"
        math_items.append((q, formatted, final_line))

    rng.shuffle(math_items)
    for idx, (q, formatted, final_line) in enumerate(math_items):
        if len(math_pre_packer.rows_tokens) >= 15500 and len(math_sft_packer.rows_tokens) >= 3200:
            break
        if len(math_pre_packer.rows_tokens) < 15500:
            ids_p, w_p = builder.encode_chatml_turn(q, formatted, sft_mode=False, high_salience_suffix=final_line)
            math_pre_packer.add_atomic_turn(ids_p, w_p)
        if idx % 4 == 0 and len(math_sft_packer.rows_tokens) < 3200:
            ids_s, w_s = builder.encode_chatml_turn(q, formatted, sft_mode=True, high_salience_suffix=final_line)
            math_sft_packer.add_atomic_turn(ids_s, w_s)

    math_pre_toks, math_pre_msks = math_pre_packer.finalize()
    math_sft_toks, math_sft_msks = math_sft_packer.finalize()
    print(
        f"      -> Stream 4 <think> Math: Pretrain={len(math_pre_toks):,} rows ({len(math_pre_toks)*SEQ_LEN/1e6:.2f}M tok) "
        f"| SFT={len(math_sft_toks):,} rows ({len(math_sft_toks)*SEQ_LEN/1e6:.2f}M tok)",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # STREAM 5: NATIVE 2K MULTI-SPAN CROSS-CHUNK ASSOCIATIVE RETRIEVAL (NIAH)
    # --------------------------------------------------------------------------
    print("[5/5] Synthesizing Non-Repeating 2K Multi-Span Associative Retrieval (1..35 token key-value gaps)...", flush=True)
    niah_pre_packer = SequenceRowPacker(builder.eos_id)
    niah_sft_packer = SequenceRowPacker(builder.eos_id)

    while len(niah_pre_packer.rows_tokens) < 15500:
        prm, queried, _ = synthesize_2k_multi_span_niah(rng)
        ids_p, w_p = builder.encode_niah_turn(prm, queried, sft_mode=False)
        if len(ids_p) <= ROW_LEN:
            niah_pre_packer.add_atomic_turn(ids_p, w_p)

    while len(niah_sft_packer.rows_tokens) < 3200:
        prm, queried, _ = synthesize_2k_multi_span_niah(rng)
        ids_s, w_s = builder.encode_niah_turn(prm, queried, sft_mode=True)
        if len(ids_s) <= ROW_LEN:
            niah_sft_packer.add_atomic_turn(ids_s, w_s)

    niah_pre_toks, niah_pre_msks = niah_pre_packer.finalize()
    niah_sft_toks, niah_sft_msks = niah_sft_packer.finalize()
    print(
        f"      -> Stream 5 2K Multi-Span NIAH: Pretrain={len(niah_pre_toks):,} rows ({len(niah_pre_toks)*SEQ_LEN/1e6:.2f}M tok) "
        f"| SFT={len(niah_sft_toks):,} rows ({len(niah_sft_toks)*SEQ_LEN/1e6:.2f}M tok)",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # SHUFFLE, SPLIT TRAIN/VAL, AND SAVE DETERMINISTIC NVME MEMMAPS
    # --------------------------------------------------------------------------
    pre_domains = {
        "edu_textbooks_prose": len(prose_toks),
        "conversational_plain_english": len(conv_pre_toks),
        "polyglot_from_scratch_code": len(code_pre_toks),
        "think_math_reasoning": len(math_pre_toks),
        "long_2k_niah_retrieval": len(niah_pre_toks),
    }
    sft_domains = {
        "conversational_plain_english": len(conv_sft_toks),
        "polyglot_from_scratch_code": len(code_sft_toks),
        "think_math_reasoning": len(math_sft_toks),
        "long_2k_niah_retrieval": len(niah_sft_toks),
    }
    for k, v in pre_domains.items():
        audit["domain_rows_pretrain"][k] = v
        audit["domain_tokens_pretrain"][k] = int(v * SEQ_LEN)
    for k, v in sft_domains.items():
        audit["domain_rows_sft"][k] = v
        audit["domain_tokens_sft"][k] = int(v * SEQ_LEN)

    all_pre_toks = np.concatenate([prose_toks, conv_pre_toks, code_pre_toks, math_pre_toks, niah_pre_toks], axis=0)
    all_pre_msks = np.concatenate([prose_msks, conv_pre_msks, code_pre_msks, math_pre_msks, niah_pre_msks], axis=0)
    perm_pre = np_rng.permutation(len(all_pre_toks))
    all_pre_toks = all_pre_toks[perm_pre]
    all_pre_msks = all_pre_msks[perm_pre]

    n_val_pre = 1500
    save_split_memmaps(out_dir, "val", all_pre_toks[:n_val_pre], all_pre_msks[:n_val_pre])
    save_split_memmaps(out_dir, "train", all_pre_toks[n_val_pre:], all_pre_msks[n_val_pre:])

    all_sft_toks = np.concatenate([conv_sft_toks, code_sft_toks, math_sft_toks, niah_sft_toks], axis=0)
    all_sft_msks = np.concatenate([conv_sft_msks, code_sft_msks, math_sft_msks, niah_sft_msks], axis=0)
    perm_sft = np_rng.permutation(len(all_sft_toks))
    all_sft_toks = all_sft_toks[perm_sft]
    all_sft_msks = all_sft_msks[perm_sft]

    n_val_sft = 400
    save_split_memmaps(out_dir, "sft_val", all_sft_toks[:n_val_sft], all_sft_msks[:n_val_sft])
    save_split_memmaps(out_dir, "sft_train", all_sft_toks[n_val_sft:], all_sft_msks[n_val_sft:])

    audit["total_pretrain_train_tokens"] = int((len(all_pre_toks) - n_val_pre) * SEQ_LEN)
    audit["total_pretrain_val_tokens"] = int(n_val_pre * SEQ_LEN)
    audit["total_sft_train_tokens"] = int((len(all_sft_toks) - n_val_sft) * SEQ_LEN)
    audit["total_sft_val_tokens"] = int(n_val_sft * SEQ_LEN)
    audit["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)

    audit_path = os.path.join(out_dir, "audit_report.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print("\n" + "=" * 96, flush=True)
    print("VAULT v2 DATA HYGIENE & BALANCE AUDIT COMPLETE!", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
