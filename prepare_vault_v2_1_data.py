#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 Philip Bartsch (batteryphil)
#
# VAULT v2.1 Data Procurement, Open-Vocabulary Binding & Decontamination Pipeline (`data_v2_1/`)
# Fixes the 3 Stage-2 data bottlenecks identified in the VAULT v2.0 adversarial audit:
#   1. Open-Vocabulary Associative Binding (>3.5M procedural compound + OOV neologism identifiers,
#      continuous 1..38 token key-value gaps, 0% lexical/subword overlap with HOLDOUT_NIAH_PREFIXES).
#   2. Prompt-Bound Polyglot Code Synthesis (.bb, C99, C++20, Python, Lua) across 150+ randomized
#      identifiers & capacities so the model obeys prompt names instead of snapping to fixed templates.
#   3. Place-Value Distributive `<think>` Arithmetic (`a * b = (tens * b) + (ones * b)`) strictly
#      holding out `45 * 4` and `"rover"` for genuine compositional arithmetic generalization.

import os
import glob
import json
import time
import random
from typing import List, Tuple, Optional

import numpy as np
import pyarrow.ipc as ipc
from transformers import AutoTokenizer

from prepare_vault_v2_data import (
    SEQ_LEN,
    ROW_LEN,
    W_IGNORE,
    W_PROMPT,
    W_PROSE,
    W_ASSIST,
    W_NEEDLE,
    HOLDOUT_SUBSTRINGS,
    HOLDOUT_NIAH_PREFIXES,
    is_clean_pair,
    generate_conceptual_prose_turn,
    parse_metamath_cot,
    VaultV2TokenBuilder,
    SequenceRowPacker,
    save_split_memmaps,
)

# Forbidden subwords from HOLDOUT_NIAH_PREFIXES so even half-stems never leak into open-vocab NIAH!
FORBIDDEN_NIAH_SUBWORDS = {
    "shield", "pod", "cryo", "lock", "warp", "core", "ion", "thruster",
    "plasma", "grid", "star", "tracker", "beacon", "mux", "grav", "sensor",
    "solar", "wing", "hull", "radar", "thermal", "sink", "dock", "clamp",
}

OPEN_PREFIXES = [
    "Cobalt", "Quartz", "Boreal", "Zenith", "Vertex", "Helix", "Strata", "Prism",
    "Aether", "Cinder", "Granite", "Obsidian", "Titanium", "Argon", "Xenon", "Krypton",
    "Rubidium", "Caesium", "Barium", "Halcyon", "Meridian", "Solstice", "Equinox", "Apogee",
    "Perigee", "Tangent", "Secant", "Spinor", "Lattice", "Conduit", "Manifold", "Aperture",
    "Fulcrum", "Gimbal", "flywheel", "Solenoid", "Inductor", "Varactor", "Thyristor", "Wavelet",
    "Cepstrum", "Bessel", "Chebyshev", "Galerkin", "Runge", "Hermite", "Laguerre", "Jacobi",
    "Sylvester", "Cayley", "Frobenius", "Noether", "Cartan", "killian", "Velox", "Ferrite",
    "Alumina", "Zircon", "Beryl", "Garnet", "Topaz", "Corundum", "Spinel", "Olivine",
]

OPEN_NOUNS = [
    "Turbine", "Manifold", "Conduit", "Transducer", "Actuator", "Modulator", "Oscillator", "Rectifier",
    "Attenuator", "Coupler", "Duplexer", "Collimator", "Interferometer", "Bolometer", "Pyrometer", "Encoder",
    "Resolver", "Tachometer", "Barometer", "Altimeter", "Gyrometer", "Magnetometer", "Fluxgate", "Preamplifier",
    "Synthesizer", "Discriminator", "Integrator", "Comparator", "Sequencer", "Scheduler", "Dispatcher", "Allocator",
    "Compactor", "Scrubber", "Validator", "Serializer", "Deserializer", "Framing", "Backplane", "Crossbar",
    "Switchgear", "Busbar", "Contactor", "Breaker", "Flywheel", "Damper", "Bellows", "Nozzle",
    "Diffuser", "Impeller", "Stator", "Armature", "Commutator", "CryostatX", " Autoclave", "Retort",
    "Condenser", "Evaporator", "Reboiler", " Absorber", "Catalyst", "Electrode", "Membrane", "Injector",
]

# Clean and verify 100% subword disjointness from HOLDOUT_NIAH_PREFIXES
OPEN_PREFIXES = [
    p.strip().capitalize() for p in OPEN_PREFIXES
    if all(sub not in p.lower() for sub in FORBIDDEN_NIAH_SUBWORDS)
]
OPEN_NOUNS = [
    n.strip().capitalize() for n in OPEN_NOUNS
    if all(sub not in n.lower() for sub in FORBIDDEN_NIAH_SUBWORDS)
]
assert len(OPEN_PREFIXES) >= 55 and len(OPEN_NOUNS) >= 55

OOV_SYLLABLES_A = ["Vel", "Krel", "Zor", "Thal", "Bryn", "Vond", "Quil", "Drax", "Myr", "Solv", "Tark", "Niv", "Phor", "Grel", "Xan", "Luv"]
OOV_SYLLABLES_B = ["va", "to", "mi", "ra", "lo", "ze", "ki", "nu", "pe", "da", "so", "ri", "fe", "ca", "vo", "lu"]
OOV_SYLLABLES_C = ["ron", "tix", "bus", "vex", "lak", "mor", "dex", "nik", "zar", "phel", "ton", "rik", "val", "gem", "syr", "dan"]


def sample_open_vocab_entity(rng: random.Random, style: str) -> str:
    while True:
        if style == "oov_neologism":
            base = rng.choice(OOV_SYLLABLES_A) + rng.choice(OOV_SYLLABLES_B) + rng.choice(OOV_SYLLABLES_C)
            tag = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 99)}"
            ent = f"{base}-{tag}" if rng.random() < 0.7 else f"{base}_{tag}"
        elif style == "snake_id":
            p = rng.choice(OPEN_PREFIXES).lower()
            n = rng.choice(OPEN_NOUNS).lower()
            ent = f"{p}_{n}_{rng.randint(1, 99):02d}"
        elif style == "dotted_service":
            p = rng.choice(OPEN_PREFIXES).lower()
            n = rng.choice(OPEN_NOUNS).lower()
            ent = f"{p}.{n}.{rng.choice('abcdefghjkmnpqrstuvwxyz')}{rng.randint(1, 9)}"
        else:
            p = rng.choice(OPEN_PREFIXES)
            n = rng.choice(OPEN_NOUNS)
            tag = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(1, 9)}"
            if rng.random() < 0.25:
                tag = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10, 99)}"
            ent = f"{p}{n}-{tag}"

        el = ent.lower()
        if all(sub not in el for sub in FORBIDDEN_NIAH_SUBWORDS) and all(h.lower() not in el for h in HOLDOUT_NIAH_PREFIXES):
            return ent


FILLER_SENTENCES_V2_1 = [
    "All redundant bus lanes completed cyclic redundancy verification with zero bit-flip events across the sampling window.",
    "Secondary harmonic dampers remained well within nominal mechanical tolerance during high-slew attitude maneuvers.",
    "Housekeeping telemetry packets were buffered in ring memory and flushed cleanly over the optical downlink channel.",
    "Ambient bus voltage ripple measured below 1.4 millivolts peak-to-peak under full multi-core vector load.",
    "Watchdog heartbeat counters incremented monotonically across every supervisory microcontroller on the backplane.",
    "Heat-pipe conductance stayed balanced across panels Alpha through Delta with zero duty-cycle throttling.",
    "ECC scrubbing engines scanned 64 megabytes of SRAM without encountering any uncorrectable multi-bit syndrome.",
    "Phase-locked oscillator drift remained below 0.03 parts per million relative to the rubidium reference clock.",
    "Deterministic time-triggered ethernet frames arrived within a 120-nanosecond jitter envelope across all switches.",
    "DMA scatter-gather descriptors retired cleanly without triggering PCIe completion timeout interrupts.",
    "Pressuremanifold transducers reported steady laminar flow across all twelve calibration orifices.",
    "Non-volatile ferroelectric FRAM shadow registers passed power-on self-test checksum validation.",
]


def synthesize_open_vocab_niah_v2_1(rng: random.Random) -> Tuple[str, List[Tuple[str, str]], bool]:
    """
    Open-vocabulary associative retrieval generator spanning 1,200..1,960 tokens with continuous
    1..36 token entity-to-value distances and zero lexical overlap with unseen holdout entities.
    Returns `(prompt, queried_pairs, single_direct_mode)`.
    """
    schema = rng.choice([
        "telemetry_manifest",
        "telemetry_manifest",
        "telemetry_mixed_gaps",
        "multifile_code",
        "json_registry",
        "prose_spec",
    ])
    style_mix = rng.choice(["camel", "oov_neologism", "snake_id", "dotted_service", "mixed"])
    n_sections = rng.randint(14, 21)

    val_kind = rng.choice(["CHK", "CHK", "VRF", "SIG", "TOK", "KEY", "HEX", "VER"])
    items: List[Tuple[str, str]] = []
    used_ents = set()
    used_vals = set()

    for _ in range(n_sections):
        st = rng.choice(["camel", "oov_neologism", "snake_id", "dotted_service"]) if style_mix == "mixed" else style_mix
        while True:
            ent = sample_open_vocab_entity(rng, st)
            if ent not in used_ents:
                used_ents.add(ent)
                break
        while True:
            if val_kind == "HEX":
                val = f"0x{rng.randint(0x1000, 0xFFFF):04X}"
            elif val_kind == "VER":
                val = f"v{rng.randint(1, 9)}.{rng.randint(10, 99)}.{rng.randint(0, 9)}"
            else:
                val = f"{val_kind}-{rng.randint(1000, 9999)}"
            if val not in used_vals:
                used_vals.add(val)
                break
        items.append((ent, val))

    n_queries = rng.choice([1, 2, 2, 2, 3])
    q_indices = sorted(rng.sample(range(len(items)), k=n_queries))
    queried = [items[i] for i in q_indices]
    single_direct = (n_queries == 1 and rng.random() < 0.35)

    def _fmt_q_names(q_list: List[Tuple[str, str]]) -> str:
        if len(q_list) == 1:
            return q_list[0][0]
        if len(q_list) == 2:
            return f"{q_list[0][0]} and {q_list[1][0]}"
        return ", ".join(e for e, _ in q_list[:-1]) + f", and {q_list[-1][0]}"

    q_names = _fmt_q_names(queried)

    if schema in ("telemetry_manifest", "telemetry_mixed_gaps"):
        blocks = ["SYSTEM TELEMETRY & VERIFICATION MANIFEST\n"]
        uniform_gap = rng.choice(["gap_15", "gap_24", "gap_4", "gap_19"])
        for idx, (ent, val) in enumerate(items, start=1):
            f1, f2 = rng.sample(FILLER_SENTENCES_V2_1, k=2)
            status = rng.choice(["NOMINAL", "OPTIMAL", "VERIFIED", "ACTIVE", "LOCKED"])
            g_mode = rng.choice(["gap_4", "gap_15", "gap_19", "gap_24", "gap_32"]) if schema == "telemetry_mixed_gaps" else uniform_gap
            if g_mode == "gap_15":
                header = f"[Subsystem {idx:02d}: {ent}] Status: {status} | Verification Code: {val}"
            elif g_mode == "gap_19":
                ch = rng.randint(0, 15)
                header = f"[Subsystem {idx:02d}: {ent}] Health: {status} | Bus Channel: {ch:02d} | Verification Code: {val}"
            elif g_mode == "gap_24":
                temp_k = rng.randint(280, 335)
                lane = rng.randint(0, 7)
                header = (
                    f"[Subsystem {idx:02d}: {ent}] Operational State: {status} | Sensor Temp: {temp_k}K "
                    f"| Link Lane: {lane} | Verification Code: {val}"
                )
            elif g_mode == "gap_32":
                temp_k = rng.randint(280, 335)
                lat_us = rng.randint(12, 95)
                header = (
                    f"[Subsystem {idx:02d}: {ent}] State: {status} | Envelope Temp: {temp_k}K "
                    f"| Roundtrip Latency: {lat_us}us | Redundancy Tier: PRIMARY | Verification Code: {val}"
                )
            else:
                header = f"[Subsystem {idx:02d}] {ent}: {val} ({status})"
            blocks.append(f"{header}\n  Log: {f1} {f2}")

        if single_direct:
            question = f"\n\nQuestion: Based strictly on the telemetry manifest above, what is the exact verification code for {q_names}?"
        else:
            question = (
                f"\n\nQuestion: Based strictly on the telemetry manifest above, what are the exact verification codes "
                f"for {q_names}? List each subsystem and its exact code."
                if len(queried) > 1
                else f"\n\nQuestion: Based strictly on the telemetry manifest above, what is the exact verification code for {q_names}? List the subsystem and its exact code."
            )
        return "\n".join(blocks) + question, queried, single_direct

    if schema == "multifile_code":
        files = []
        for idx, (ent, val) in enumerate(items, start=1):
            ident = ent.replace("-", "_").replace(".", "_").upper()
            slug = ent.replace("-", "_").replace(".", "_").lower()
            f1 = rng.choice(FILLER_SENTENCES_V2_1)
            files.append(
                f"<file_sep>include/unit_{idx:02d}_{slug}.h\n"
                f"// Component: {ent} — {f1}\n"
                f"#pragma once\n"
                f"#define {ident}_SLOT {idx}\n"
                f"#define {ident}_VERIFY_TAG \"{val}\"\n"
                f"static inline int probe_{slug}(void) {{ return {idx}; }}\n"
            )
        question = (
            f"\n\nQuestion: Across the header files above, what exact `VERIFY_TAG` values are defined for {q_names}?"
        )
        return "\n".join(files) + question, queried, single_direct

    if schema == "json_registry":
        entries_txt = ["SERVICE CONFIGURATION REGISTRY\n["]
        for idx, (ent, val) in enumerate(items, start=1):
            f1 = rng.choice(FILLER_SENTENCES_V2_1)
            region = rng.choice(["us-east", "eu-central", "ap-north", "orbital-leo"])
            comma = "," if idx < len(items) else ""
            entries_txt.append(
                f'  {{"id": "{ent}", "seq": {idx}, "zone": "{region}", "notes": "{f1}", "auth_token": "{val}"}}{comma}'
            )
        entries_txt.append("]")
        question = (
            f"\n\nQuestion: From the JSON service registry above, report the exact `auth_token` assigned to {q_names}."
        )
        return "\n".join(entries_txt) + question, queried, single_direct

    paras = ["QUALIFICATION & CALIBRATION DOSSIER\n"]
    for idx, (ent, val) in enumerate(items, start=1):
        f1 = rng.choice(FILLER_SENTENCES_V2_1)
        cycles = rng.randint(12, 64)
        paras.append(
            f"Record {idx:02d}: During acceptance testing, assembly {ent} completed {cycles} continuous vacuum "
            f"endurance cycles with zero bus arbitration faults and received calibration token {val}. {f1}"
        )
    question = (
        f"\n\nQuestion: According to the dossier above, what exact calibration tokens were assigned to {q_names}?"
    )
    return "\n\n".join(paras) + question, queried, single_direct


# ==============================================================================
# 2. PROMPT-BOUND POLYGLOT CODE GENERATORS (ZERO SNAP-TO, 100% DECONTAMINATED)
# ==============================================================================

BB_STRUCT_NAMES = [
    "WavePacket", "RigidBody", "AudioVoice", "BoidAgent", "FluidParcel", "LatticeSite",
    "SpringMass", "ClothVertex", "WaypointNode", "RaySample", "VortexRing", "ElasticRod",
    "CarrierPulse", "DipoleMoment", "FieldCell", "MeshPoint", "OrbitState", "PendulumArm",
    "HarmonicMode", "NavSample", "KineticGrain", "AcousticNode", "PhaseRotor", "FluxTube",
]

CPP_TEMPLATE_NAMES = [
    "BoundedLifo", "StaticVault", "InlineShelf", "ArrayPocket", "CappedTray", "SlotLedger",
    "RingChannel", "CircularPipe", "BoundedQueue", "StaticDeque", "SlabWindow", "RingBuffer",
    "InlineBuffer", "FixedLedger", "CapacityBox", "CompactFifo", "TokenBucket", "FrameRing",
]

LUA_CLASS_NAMES = [
    "Oscillator", "SpringDamper", "RateLimiter", "CounterMetric", "RotorBody", "CooldownTimer",
    "TweenCurve", "SignalFilter", "CameraRig", "MotorJoint", "EnergyPool", "PathFollower",
    "WaveEmitter", "StateTracker", "PidLoop", "HealthMeter", "DriftEstimator", "ClockDivider",
]


def generate_prompt_bound_polyglot_turn(rng: random.Random) -> Tuple[str, str]:
    lang = rng.choice(["buildbit", "c99", "cpp20", "python", "lua"])

    if lang == "buildbit":
        sname = rng.choice(BB_STRUCT_NAMES)
        field_sets = [
            ([("x", "f32"), ("vx", "f32"), ("ax", "f32")], "x", "vx", "ax"),
            ([("pos", "f32"), ("vel", "f32"), ("acc", "f32")], "pos", "vel", "acc"),
            ([("u", "f32"), ("du", "f32"), ("d2u", "f32")], "u", "du", "d2u"),
            ([("px", "f32"), ("py", "f32"), ("vx", "f32"), ("vy", "f32")], "px", "vx", None),
            ([("phase", "f32"), ("freq", "f32"), ("gain", "f32")], "phase", "freq", None),
            ([("energy", "f32"), ("flux", "f32"), ("damping", "f32")], "energy", "flux", None),
        ]
        fields, f_dst, f_vel, f_acc = rng.choice(field_sets)
        count = rng.choice([1024, 2048, 8192, 16384, 32768, 65536])
        arena_mb = rng.choice([8, 16, 32, 64])
        # Strictly avoid combining 4096 with anything resembling Photon (Photon is already excluded from BB_STRUCT_NAMES)
        arena_bytes = arena_mb * 1048576
        dt = rng.choice(["0.01", "0.016", "0.005", "0.025"])
        f_decl = ", ".join(f"{fn}: {ft}" for fn, ft in fields)
        fn_list_prose = ", ".join(f"`{fn}`" for fn, _ in fields[:-1]) + f", and `{fields[-1][0]}`"

        if f_acc is not None and rng.random() < 0.55:
            prompt = (
                f"Write a BuildBit (.bb) function that defines `soa struct {sname}` with `f32` fields {fn_list_prose}, "
                f"allocates {count} elements in a {arena_mb} MB `gpu_arena`, and updates `{f_dst}` and `{f_vel}` in parallel."
            )
            body = (
                f"        items[i].{f_vel} = items[i].{f_vel} + items[i].{f_acc} * {dt}\n"
                f"        items[i].{f_dst} = items[i].{f_dst} + items[i].{f_vel} * {dt}"
            )
        else:
            prompt = (
                f"Write a BuildBit (.bb) program that defines `soa struct {sname}` with fields `{f_decl}`, "
                f"allocates {count} elements in a {arena_mb} MB `gpu_arena`, and updates `{f_dst}` in a `parallel for` loop."
            )
            body = f"        items[i].{f_dst} = items[i].{f_dst} + items[i].{f_vel} * {dt}"

        code = (
            f"```rust\n"
            f"soa struct {sname} {{\n"
            f"    {f_decl}\n"
            f"}}\n\n"
            f"fn update_{sname.lower()}() {{\n"
            f"    gpu_arena arena = {arena_bytes}\n"
            f"    let items = alloc_soa(arena, {sname}, {count})\n"
            f"    parallel for i in 0..{count} {{\n"
            f"{body}\n"
            f"    }}\n"
            f"    reset(arena)\n"
            f"}}\n"
            f"```"
        )
        return prompt, code

    if lang == "c99":
        variant = rng.choice(["reverse_non_i32", "partition", "dedup", "prefix_sum", "clamp", "dot", "bsearch", "axpy"])
        if variant == "reverse_non_i32":
            # Teaches clean two-pointer in-place swap on bytes, 16-bit unsigned, 64-bit, or doubles (NEVER 32-bit int!)
            ctype, label, fname = rng.choice([
                ("double", "double-precision floats", "reverse_f64_slice"),
                ("uint8_t", "bytes (`uint8_t`)", "reverse_bytes_inplace"),
                ("int16_t", "16-bit signed integers", "reverse_i16_buf"),
                ("int64_t", "64-bit integers (`int64_t`)", "reverse_i64_span"),
            ])
            hdr = "#include <stdint.h>\n#include <stddef.h>" if "int" in ctype else "#include <stddef.h>"
            prompt = f"Write a C99 function `{fname}` to reverse an array of {label} in place."
            code = (
                f"```c\n{hdr}\n\n"
                f"void {fname}({ctype} *arr, size_t n) {{\n"
                f"    if (!arr || n < 2) return;\n"
                f"    size_t lo = 0;\n"
                f"    size_t hi = n - 1;\n"
                f"    while (lo < hi) {{\n"
                f"        {ctype} tmp = arr[lo];\n"
                f"        arr[lo] = arr[hi];\n"
                f"        arr[hi] = tmp;\n"
                f"        ++lo;\n"
                f"        --hi;\n"
                f"    }}\n"
                f"}}\n```"
            )
            return prompt, code
        if variant == "partition":
            fname = rng.choice(["partition_i32_pivot", "split_below_threshold", "hoare_partition_f32"])
            ctype = "float" if "f32" in fname else "int32_t"
            hdr = "#include <stddef.h>" if ctype == "float" else "#include <stdint.h>\n#include <stddef.h>"
            prompt = f"Write a C99 function `{fname}` that partitions a `{ctype}` array in place so elements `< pivot` come first, returning the split index."
            code = (
                f"```c\n{hdr}\n\n"
                f"size_t {fname}({ctype} *arr, size_t n, {ctype} pivot) {{\n"
                f"    if (!arr) return 0;\n"
                f"    size_t write_idx = 0;\n"
                f"    for (size_t i = 0; i < n; ++i) {{\n"
                f"        if (arr[i] < pivot) {{\n"
                f"            {ctype} tmp = arr[write_idx];\n"
                f"            arr[write_idx] = arr[i];\n"
                f"            arr[i] = tmp;\n"
                f"            ++write_idx;\n"
                f"        }}\n"
                f"    }}\n"
                f"    return write_idx;\n"
                f"}}\n```"
            )
            return prompt, code
        if variant == "dedup":
            prompt = "Write a C99 function `dedup_sorted_i32` that removes duplicate values from a sorted `int32_t` array in place and returns the new length."
            code = (
                "```c\n#include <stdint.h>\n#include <stddef.h>\n\n"
                "size_t dedup_sorted_i32(int32_t *arr, size_t n) {\n"
                "    if (!arr || n == 0) return 0;\n"
                "    size_t out = 1;\n"
                "    for (size_t i = 1; i < n; ++i) {\n"
                "        if (arr[i] != arr[out - 1]) {\n"
                "            arr[out++] = arr[i];\n"
                "        }\n"
                "    }\n"
                "    return out;\n"
                "}\n```"
            )
            return prompt, code
        if variant == "prefix_sum":
            prompt = "Write a C99 function `inclusive_scan_i64` that computes the in-place running prefix sum of an `int64_t` array."
            code = (
                "```c\n#include <stdint.h>\n#include <stddef.h>\n\n"
                "void inclusive_scan_i64(int64_t *arr, size_t n) {\n"
                "    if (!arr) return;\n"
                "    for (size_t i = 1; i < n; ++i) {\n"
                "        arr[i] += arr[i - 1];\n"
                "    }\n"
                "}\n```"
            )
            return prompt, code
        if variant == "clamp":
            prompt = "Write a C99 function `clamp_f32_array` that clamps each element of a `float` buffer to `[min_val, max_val]` in place."
            code = (
                "```c\n#include <stddef.h>\n\n"
                "void clamp_f32_array(float * restrict buf, size_t n, float min_val, float max_val) {\n"
                "    if (!buf) return;\n"
                "    for (size_t i = 0; i < n; ++i) {\n"
                "        float x = buf[i];\n"
                "        if (x < min_val) x = min_val;\n"
                "        else if (x > max_val) x = max_val;\n"
                "        buf[i] = x;\n"
                "    }\n"
                "}\n```"
            )
            return prompt, code
        if variant == "dot":
            prompt = "Write a C99 function `dot_f32` that computes the inner product of two `const float *` vectors."
            code = (
                "```c\n#include <stddef.h>\n\n"
                "float dot_f32(const float * restrict a, const float * restrict b, size_t n) {\n"
                "    float sum = 0.0f;\n"
                "    for (size_t i = 0; i < n; ++i) {\n"
                "        sum += a[i] * b[i];\n"
                "    }\n"
                "    return sum;\n"
                "}\n```"
            )
            return prompt, code
        if variant == "axpy":
            prompt = "Write a C99 function `saxpy_f32` that computes `y[i] = alpha * x[i] + y[i]` over `n` floats."
            code = (
                "```c\n#include <stddef.h>\n\n"
                "void saxpy_f32(size_t n, float alpha, const float * restrict x, float * restrict y) {\n"
                "    if (!x || !y) return;\n"
                "    for (size_t i = 0; i < n; ++i) {\n"
                "        y[i] += alpha * x[i];\n"
                "    }\n"
                "}\n```"
            )
            return prompt, code
        prompt = "Write a C99 function `binary_search_i32` that finds `key` in a sorted `int32_t` array and returns its index or `-1`."
        code = (
            "```c\n#include <stdint.h>\n#include <stddef.h>\n\n"
            "ptrdiff_t binary_search_i32(const int32_t *arr, size_t n, int32_t key) {\n"
            "    size_t lo = 0, hi = n;\n"
            "    while (lo < hi) {\n"
            "        size_t mid = lo + ((hi - lo) >> 1);\n"
            "        if (arr[mid] == key) return (ptrdiff_t)mid;\n"
            "        if (arr[mid] < key) lo = mid + 1;\n"
            "        else hi = mid;\n"
            "    }\n"
            "    return -1;\n"
            "}\n```"
        )
        return prompt, code

    if lang == "cpp20":
        cname = rng.choice(CPP_TEMPLATE_NAMES)
        cap = rng.choice([16, 32, 64, 96, 192, 256, 512, 1024])
        ds_kind = rng.choice(["lifo", "lifo", "fifo", "static_vec"])
        if ds_kind == "lifo":
            prompt = f"Write a C++20 generic class template `{cname}<T, {cap}>` implementing a fixed-capacity LIFO container with `push`, `pop`, and `size` methods."
            code = (
                f"```cpp\n#include <array>\n#include <cstddef>\n#include <stdexcept>\n\n"
                f"template <typename T, std::size_t Capacity = {cap}>\n"
                f"class {cname} {{\n"
                f"    std::array<T, Capacity> data_{{}};\n"
                f"    std::size_t top_ = 0;\n\n"
                f"public:\n"
                f"    constexpr void push(const T& value) {{\n"
                f"        if (top_ >= Capacity) throw std::overflow_error(\"{cname} overflow\");\n"
                f"        data_[top_++] = value;\n"
                f"    }}\n\n"
                f"    constexpr T pop() {{\n"
                f"        if (top_ == 0) throw std::underflow_error(\"{cname} underflow\");\n"
                f"        return data_[--top_];\n"
                f"    }}\n\n"
                f"    [[nodiscard]] constexpr std::size_t size() const noexcept {{\n"
                f"        return top_;\n"
                f"    }}\n"
                f"}};\n```"
            )
            return prompt, code
        if ds_kind == "static_vec":
            prompt = f"Write a C++20 generic class template `{cname}<T, {cap}>` with `append`, `back`, `pop_back`, and `size` methods."
            code = (
                f"```cpp\n#include <array>\n#include <cstddef>\n#include <stdexcept>\n\n"
                f"template <typename T, std::size_t Capacity = {cap}>\n"
                f"class {cname} {{\n"
                f"    std::array<T, Capacity> items_{{}};\n"
                f"    std::size_t len_ = 0;\n\n"
                f"public:\n"
                f"    constexpr void append(const T& item) {{\n"
                f"        if (len_ >= Capacity) throw std::length_error(\"{cname} full\");\n"
                f"        items_[len_++] = item;\n"
                f"    }}\n\n"
                f"    constexpr const T& back() const {{\n"
                f"        if (len_ == 0) throw std::out_of_range(\"{cname} empty\");\n"
                f"        return items_[len_ - 1];\n"
                f"    }}\n\n"
                f"    constexpr void pop_back() {{\n"
                f"        if (len_ == 0) throw std::out_of_range(\"{cname} empty\");\n"
                f"        --len_;\n"
                f"    }}\n\n"
                f"    [[nodiscard]] constexpr std::size_t size() const noexcept {{ return len_; }}\n"
                f"}};\n```"
            )
            return prompt, code
        prompt = f"Write a C++20 generic class template `{cname}<T, {cap}>` implementing a bounded FIFO ring queue with `push`, `pop`, and `size` methods."
        code = (
            f"```cpp\n#include <array>\n#include <cstddef>\n#include <stdexcept>\n\n"
            f"template <typename T, std::size_t Capacity = {cap}>\n"
            f"class {cname} {{\n"
            f"    std::array<T, Capacity> ring_{{}};\n"
            f"    std::size_t head_ = 0;\n"
            f"    std::size_t tail_ = 0;\n"
            f"    std::size_t count_ = 0;\n\n"
            f"public:\n"
            f"    constexpr bool push(const T& item) noexcept {{\n"
            f"        if (count_ == Capacity) return false;\n"
            f"        ring_[tail_] = item;\n"
            f"        tail_ = (tail_ + 1) % Capacity;\n"
            f"        ++count_;\n"
            f"        return true;\n"
            f"    }}\n\n"
            f"    constexpr T pop() {{\n"
            f"        if (count_ == 0) throw std::underflow_error(\"{cname} empty\");\n"
            f"        T out = ring_[head_];\n"
            f"        head_ = (head_ + 1) % Capacity;\n"
            f"        --count_;\n"
            f"        return out;\n"
            f"    }}\n\n"
            f"    [[nodiscard]] constexpr std::size_t size() const noexcept {{ return count_; }}\n"
            f"}};\n```"
        )
        return prompt, code

    if lang == "lua":
        cname = rng.choice(LUA_CLASS_NAMES)
        behaviors = [
            ("x, y, vx, vy", "self.x = x or 0.0\n    self.y = y or 0.0\n    self.vx = vx or 0.0\n    self.vy = vy or 0.0", "step(dt)", "self.x = self.x + self.vx * dt\n    self.y = self.y + self.vy * dt"),
            ("angle, omega", "self.angle = angle or 0.0\n    self.omega = omega or 1.0", "advance(dt)", "self.angle = self.angle + self.omega * dt"),
            ("value, decay_rate", "self.value = value or 1.0\n    self.decay_rate = decay_rate or 0.5", "tick(dt)", "self.value = math.max(0.0, self.value - self.decay_rate * dt)"),
            ("pos, target, gain", "self.pos = pos or 0.0\n    self.target = target or 0.0\n    self.gain = gain or 4.0", "integrate(dt)", "self.pos = self.pos + (self.target - self.pos) * self.gain * dt"),
        ]
        ctor_args, ctor_body, m_sig, m_body = rng.choice(behaviors)
        prompt = f"Write a Lua module implementing a `{cname}` class using metatables (`__index` and `:{m_sig}`)."
        code = (
            f"```lua\nlocal {cname} = {{}}\n{cname}.__index = {cname}\n\n"
            f"function {cname}.new({ctor_args})\n"
            f"    local self = setmetatable({{}}, {cname})\n"
            f"    {ctor_body}\n"
            f"    return self\n"
            f"end\n\n"
            f"function {cname}:{m_sig}\n"
            f"    {m_body}\n"
            f"    return self\n"
            f"end\n\n"
            f"return {cname}\n```"
        )
        return prompt, code

    py_tasks = [
        (
            "[python] Write a Python function to find the minimum element and arithmetic mean of a non-empty list of numbers.",
            "```python\ndef min_and_mean(nums: list[float]) -> tuple[float, float]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    minimum = nums[0]\n    total = 0.0\n    for x in nums:\n        if x < minimum:\n            minimum = x\n        total += x\n    return minimum, total / len(nums)\n```",
        ),
        (
            "[python] Write a Python function to compute the minimum, maximum, and range span `(max - min)` of a list of numbers.",
            "```python\ndef min_max_span(nums: list[float]) -> tuple[float, float, float]:\n    if not nums:\n        raise ValueError('nums must not be empty')\n    lo = hi = nums[0]\n    for x in nums[1:]:\n        if x < lo:\n            lo = x\n        elif x > hi:\n            hi = x\n    return lo, hi, hi - lo\n```",
        ),
        (
            "[python] Write a Python function to count the positive numbers and compute the product of all elements in a list.",
            "```python\ndef count_positive_and_product(nums: list[int]) -> tuple[int, int]:\n    pos_count = 0\n    prod = 1\n    for x in nums:\n        if x > 0:\n            pos_count += 1\n        prod *= x\n    return pos_count, prod\n```",
        ),
        (
            "[python] Write a Python function `merge_intervals` that merges overlapping `[start, end]` intervals.",
            "```python\ndef merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    if not intervals:\n        return []\n    ordered = sorted(intervals, key=lambda item: item[0])\n    merged = [ordered[0][:]]\n    for start, end in ordered[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged\n```",
        ),
        (
            "[python] Write a Python function `two_sum_indices` that returns the indices `(i, j)` of two numbers in `nums` that add up to `target`.",
            "```python\ndef two_sum_indices(nums: list[int], target: int) -> tuple[int, int] | None:\n    seen: dict[int, int] = {}\n    for idx, val in enumerate(nums):\n        need = target - val\n        if need in seen:\n            return seen[need], idx\n        seen[val] = idx\n    return None\n```",
        ),
    ]
    return rng.choice(py_tasks)


# ==============================================================================
# 3. PLACE-VALUE DISTRIBUTIVE <think> ARITHMETIC GENERATOR (HOLDS OUT 45 * 4!)
# ==============================================================================

def _place_value_mul_line(a: int, b: int) -> str:
    hundreds = (a // 100) * 100
    tens = ((a % 100) // 10) * 10
    ones = a % 10
    parts = [p for p in (hundreds, tens, ones) if p > 0]
    if len(parts) <= 1 and b < 12:
        return f"Compute {a} * {b} = {a * b}."
    if len(parts) >= 2 and b <= 12:
        lhs = " + ".join(f"({p} * {b})" for p in parts)
        mid = " + ".join(str(p * b) for p in parts)
        return f"Break {a} into {' + '.join(str(p) for p in parts)}: {a} * {b} = {lhs} = {mid} = {a * b}."
    # Also decompose b if b >= 12
    b_tens = (b // 10) * 10
    b_ones = b % 10
    if b_ones > 0:
        return (
            f"Break {b} into {b_tens} + {b_ones}: "
            f"{a} * {b} = ({a} * {b_tens}) + ({a} * {b_ones}) = {a * b_tens} + {a * b_ones} = {a * b}."
        )
    return f"Compute {a} * {b} = {a * b}."


def generate_decomposed_math_turn(rng: random.Random) -> Tuple[str, str, str]:
    """Place-value decomposed `<think>` word problems strictly holding out `45 * 4` and `'rover'`."""
    kind = rng.choice(["kinematics", "kinematics", "energy", "throughput", "inventory_add", "inventory_sub"])

    if kind == "kinematics":
        veh = rng.choice(["probe", "freight train", "autonomous glider", "courier van", "survey vessel", "lunar cart"])
        while True:
            speed = rng.randint(14, 96)
            hours = rng.randint(3, 12)
            if not (speed == 45 and hours == 4) and not (speed == 4 and hours == 45):
                break
        dist = speed * hours
        prompt = rng.choice([
            f"If a {veh} travels at {speed} miles per hour for {hours} hours, how far does it travel in total?",
            f"A {veh} moves at a steady speed of {speed} kilometers per hour for {hours} hours. What total distance does it cover?",
        ])
        unit = "miles" if "miles" in prompt else "kilometers"
        think = (
            f"Distance equals speed multiplied by time ({speed} * {hours}).\n"
            f"{_place_value_mul_line(speed, hours)}"
        )
        ans = f"The {veh} travels {dist} {unit} in total."
        return prompt, think, ans

    if kind == "energy":
        watts = rng.choice([35, 45, 55, 65, 75, 85, 95, 115, 135, 165, 225])
        hrs = rng.choice([3, 4, 5, 6, 7, 8, 9])
        if watts == 45 and hrs == 4:
            hrs = 6
        wh = watts * hrs
        prompt = f"A sensor array draws {watts} watts continuously for {hrs} hours. How many watt-hours of energy does it use?"
        think = (
            f"Energy equals power multiplied by time ({watts} * {hrs}).\n"
            f"{_place_value_mul_line(watts, hrs)}"
        )
        ans = f"It uses {wh} watt-hours of energy."
        return prompt, think, ans

    if kind == "throughput":
        rate = rng.randint(22, 145)
        mins = rng.randint(3, 11)
        if rate == 45 and mins == 4:
            mins = 7
        tot = rate * mins
        prompt = f"A compiler worker builds {rate} modules per minute. How many modules does it build in {mins} minutes?"
        think = (
            f"Multiply the per-minute rate by the number of minutes ({rate} * {mins}).\n"
            f"{_place_value_mul_line(rate, mins)}"
        )
        ans = f"It builds {tot} modules in {mins} minutes."
        return prompt, think, ans

    crates = rng.randint(12, 48)
    per_crate = rng.randint(4, 16)
    if crates == 45 and per_crate == 4:
        crates = 46
    delta = rng.randint(7, 39)
    base = crates * per_crate
    if kind == "inventory_add":
        tot = base + delta
        prompt = f"A lab stores {crates} trays with {per_crate} optical filters per tray, plus {delta} loose filters on a bench. How many filters are there in total?"
        think = (
            f"First compute filters in trays:\n"
            f"{_place_value_mul_line(crates, per_crate)}\n"
            f"Add the {delta} loose filters: {base} + {delta} = {tot}."
        )
        ans = f"There are {tot} optical filters in total."
        return prompt, think, ans
    else:
        used = min(delta, base - 5)
        rem = base - used
        prompt = f"A technician unpacks {crates} boxes containing {per_crate} bolts each and uses {used} bolts during assembly. How many bolts remain?"
        think = (
            f"First compute total unpacked bolts:\n"
            f"{_place_value_mul_line(crates, per_crate)}\n"
            f"Subtract the {used} bolts used: {base} - {used} = {rem}."
        )
        ans = f"There are {rem} bolts remaining."
        return prompt, think, ans


def encode_open_niah_turn(
    builder: VaultV2TokenBuilder,
    prompt: str,
    queried: List[Tuple[str, str]],
    single_direct: bool,
    sft_mode: bool,
) -> Tuple[List[int], List[int]]:
    if single_direct and len(queried) == 1:
        ent, val = queried[0]
        resp = f"The exact code for {ent} is {val}."
        return builder.encode_chatml_turn(prompt, resp, sft_mode=sft_mode, high_salience_suffix=f"{val}.")
    return builder.encode_niah_turn(prompt, queried, sft_mode=sft_mode)


def main():
    t0 = time.perf_counter()
    rng = random.Random(20261030)
    np_rng = np.random.default_rng(20261030)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data_v2", "raw")
    out_dir = os.path.join(base_dir, "data_v2_1")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", local_files_only=True)
    builder = VaultV2TokenBuilder(tok)

    print("=" * 96, flush=True)
    print("VAULT v2.1 OPEN-VOCABULARY / PROMPT-BOUND / DISTRIBUTIVE-COT DATA BUILDER", flush=True)
    print("=" * 96, flush=True)

    audit = {
        "rejected_benchmark_holdout": 0,
        "rejected_code_review_poison": 0,
        "adapt_rows_by_domain": {},
        "sft_rows_by_domain": {},
    }

    # 1. Open-Vocabulary Multi-Span Associative Retrieval (NIAH)
    print("[1/4] Synthesizing Open-Vocabulary Multi-Span NIAH (1..36 tok gaps, >3.5M identifier space)...", flush=True)
    niah_adapt = SequenceRowPacker(builder.eos_id)
    niah_sft = SequenceRowPacker(builder.eos_id)

    while len(niah_adapt.rows_tokens) < 5200:
        prm, queried, single_direct = synthesize_open_vocab_niah_v2_1(rng)
        assert all(h not in prm.lower() for h in HOLDOUT_SUBSTRINGS)
        ids, w = encode_open_niah_turn(builder, prm, queried, single_direct, sft_mode=False)
        if len(ids) <= ROW_LEN:
            niah_adapt.add_atomic_turn(ids, w)

    while len(niah_sft.rows_tokens) < 3000:
        prm, queried, single_direct = synthesize_open_vocab_niah_v2_1(rng)
        assert all(h not in prm.lower() for h in HOLDOUT_SUBSTRINGS)
        ids, w = encode_open_niah_turn(builder, prm, queried, single_direct, sft_mode=True)
        if len(ids) <= ROW_LEN:
            niah_sft.add_atomic_turn(ids, w)

    niah_adapt_t, niah_adapt_m = niah_adapt.finalize()
    niah_sft_t, niah_sft_m = niah_sft.finalize()
    print(f"      -> Open-Vocab NIAH: Adapt={len(niah_adapt_t):,} rows | SFT={len(niah_sft_t):,} rows", flush=True)

    # 2. Prompt-Bound Polyglot From-Scratch Code (.bb, C99, C++20, Python, Lua + Public Clean Code)
    print("[2/4] Building Prompt-Bound Polyglot Code (.bb + C99 + C++20 + Python + Lua + Magicoder/CF)...", flush=True)
    code_adapt = SequenceRowPacker(builder.eos_id)
    code_sft = SequenceRowPacker(builder.eos_id)

    # Authentic BuildBit (.bb) repository files
    bb_files = sorted(
        set(
            glob.glob(os.path.join(base_dir, "..", "buildbit", "**", "*.bb"), recursive=True)
            + glob.glob(os.path.join(base_dir, "..", "buildbit-os", "**", "*.bb"), recursive=True)
        )
    )
    for _ in range(4):
        rng.shuffle(bb_files)
        for bf in bb_files:
            if os.path.isfile(bf):
                with open(bf, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if len(content) >= 80 and "soa struct photon" not in content.lower():
                    ids = tok.encode(f"<file_sep>{os.path.basename(bf)}\n{content}", add_special_tokens=False)
                    code_adapt.add_continuous_doc(ids, w_val=W_PROSE)

    code_pairs: List[Tuple[str, str]] = []
    magi_path = os.path.join(raw_dir, "ise-uiuc__Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl")
    with open(magi_path, "r", encoding="utf-8") as f:
        for idx_l, line in enumerate(f):
            if idx_l >= 18000:
                break
            r = json.loads(line)
            ok, reason = is_clean_pair(r.get("problem", ""), r.get("solution", ""))
            if not ok:
                if reason == "benchmark_holdout":
                    audit["rejected_benchmark_holdout"] += 1
                elif reason == "code_review_poison":
                    audit["rejected_code_review_poison"] += 1
                continue
            code_pairs.append((r["problem"], r["solution"]))

    cf_path = os.path.join(raw_dir, "m-a-p__CodeFeedback-Filtered-Instruction", "CodeFeedback-Filtered-Instruction.jsonl")
    with open(cf_path, "r", encoding="utf-8") as f:
        for idx_l, line in enumerate(f):
            if idx_l >= 16000:
                break
            r = json.loads(line)
            ok, reason = is_clean_pair(r.get("query", ""), r.get("answer", ""))
            if not ok:
                continue
            code_pairs.append((r["query"], r["answer"]))

    # 16,000 prompt-bound diverse identifier/capacity pairs across .bb, C99, C++20, Python, Lua
    for _ in range(16000):
        u_p, a_p = generate_prompt_bound_polyglot_turn(rng)
        ok, reason = is_clean_pair(u_p, a_p)
        assert ok, f"Procedural code pair failed hygiene ({reason}): {u_p}"
        code_pairs.append((u_p, a_p))

    rng.shuffle(code_pairs)
    for idx, (u_txt, a_txt) in enumerate(code_pairs):
        if len(code_adapt.rows_tokens) >= 4800 and len(code_sft.rows_tokens) >= 3000:
            break
        if len(code_adapt.rows_tokens) < 4800:
            ids_p, w_p = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=False)
            code_adapt.add_atomic_turn(ids_p, w_p)
        if idx % 2 == 0 and len(code_sft.rows_tokens) < 3000:
            ids_s, w_s = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            code_sft.add_atomic_turn(ids_s, w_s)

    code_adapt_t, code_adapt_m = code_adapt.finalize()
    code_sft_t, code_sft_m = code_sft.finalize()
    print(f"      -> Prompt-Bound Code: Adapt={len(code_adapt_t):,} rows | SFT={len(code_sft_t):,} rows", flush=True)

    # 3. Place-Value Distributive <think> Math + MetaMathQA
    print("[3/4] Building Distributive Place-Value <think> Math + MetaMathQA...", flush=True)
    math_adapt = SequenceRowPacker(builder.eos_id)
    math_sft = SequenceRowPacker(builder.eos_id)

    math_items: List[Tuple[str, str, str]] = []
    mm_path = os.path.join(raw_dir, "meta-math__MetaMathQA", "MetaMathQA-395K.json")
    with open(mm_path, "r", encoding="utf-8") as f:
        mm_data = json.load(f)
    rng.shuffle(mm_data)
    for r in mm_data[:22000]:
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

    for _ in range(14000):
        q, steps, final_line = generate_decomposed_math_turn(rng)
        formatted = f"<think>\n{steps}\n</think>\n{final_line}"
        ok, reason = is_clean_pair(q, formatted)
        assert ok, f"Math hygiene violation ({reason}): {q}"
        math_items.append((q, formatted, final_line))

    rng.shuffle(math_items)
    for idx, (q, formatted, final_line) in enumerate(math_items):
        if len(math_adapt.rows_tokens) >= 2800 and len(math_sft.rows_tokens) >= 1800:
            break
        if len(math_adapt.rows_tokens) < 2800:
            ids_p, w_p = builder.encode_chatml_turn(q, formatted, sft_mode=False, high_salience_suffix=final_line)
            math_adapt.add_atomic_turn(ids_p, w_p)
        if idx % 2 == 0 and len(math_sft.rows_tokens) < 1800:
            ids_s, w_s = builder.encode_chatml_turn(q, formatted, sft_mode=True, high_salience_suffix=final_line)
            math_sft.add_atomic_turn(ids_s, w_s)

    math_adapt_t, math_adapt_m = math_adapt.finalize()
    math_sft_t, math_sft_m = math_sft.finalize()
    print(f"      -> Distributive <think> Math: Adapt={len(math_adapt_t):,} rows | SFT={len(math_sft_t):,} rows", flush=True)

    # 4. Plain-English Conversational & CS Explanations (Zero <think>, Zero Code Fences)
    print("[4/4] Building Clean Plain-English Conversational & CS Explanations...", flush=True)
    conv_adapt = SequenceRowPacker(builder.eos_id)
    conv_sft = SequenceRowPacker(builder.eos_id)

    allowed_cats = {"information-seeking", "reasoning", "advice-seeking", "planning", "brainstorming", "data-analysis"}
    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    smoltalk_arrows = sorted(glob.glob(os.path.join(hf_cache, "datasets/HuggingFaceTB___smoltalk/**/*train-*.arrow"), recursive=True))
    conv_pairs: List[Tuple[str, str]] = []

    for ap in smoltalk_arrows:
        if len(conv_pairs) >= 18000:
            break
        with ipc.open_stream(ap) as reader:
            tbl = reader.read_all()
            for msgs, cat, qual in zip(tbl["messages"].to_pylist(), tbl["category"].to_pylist(), tbl["quality"].to_pylist()):
                if qual not in ("excellent", "good") or cat not in allowed_cats or not msgs or len(msgs) < 2:
                    continue
                u_msg = msgs[0].get("content", "") if msgs[0].get("role") == "user" else ""
                a_msg = msgs[1].get("content", "") if msgs[1].get("role") == "assistant" else ""
                if "```" in a_msg or "<think>" in a_msg:
                    continue
                ok, _ = is_clean_pair(u_msg, a_msg)
                if ok:
                    conv_pairs.append((u_msg, a_msg))
                    if len(conv_pairs) >= 18000:
                        break

    for _ in range(2500):
        conv_pairs.append(generate_conceptual_prose_turn(rng))

    rng.shuffle(conv_pairs)
    for idx, (u_txt, a_txt) in enumerate(conv_pairs):
        if len(conv_adapt.rows_tokens) >= 2200 and len(conv_sft.rows_tokens) >= 1400:
            break
        if len(conv_adapt.rows_tokens) < 2200:
            ids_p, w_p = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=False)
            conv_adapt.add_atomic_turn(ids_p, w_p)
        if idx % 2 == 0 and len(conv_sft.rows_tokens) < 1400:
            ids_s, w_s = builder.encode_chatml_turn(u_txt, a_txt, sft_mode=True)
            conv_sft.add_atomic_turn(ids_s, w_s)

    conv_adapt_t, conv_adapt_m = conv_adapt.finalize()
    conv_sft_t, conv_sft_m = conv_sft.finalize()
    print(f"      -> Plain-English Q&A: Adapt={len(conv_adapt_t):,} rows | SFT={len(conv_sft_t):,} rows", flush=True)

    # Combine, shuffle, split train/val, and write memmaps
    audit["adapt_rows_by_domain"] = {
        "open_vocab_niah": int(len(niah_adapt_t)),
        "prompt_bound_polyglot_code": int(len(code_adapt_t)),
        "distributive_think_math": int(len(math_adapt_t)),
        "plain_english_qa": int(len(conv_adapt_t)),
    }
    audit["sft_rows_by_domain"] = {
        "open_vocab_niah": int(len(niah_sft_t)),
        "prompt_bound_polyglot_code": int(len(code_sft_t)),
        "distributive_think_math": int(len(math_sft_t)),
        "plain_english_qa": int(len(conv_sft_t)),
    }

    all_adapt_t = np.concatenate([niah_adapt_t, code_adapt_t, math_adapt_t, conv_adapt_t], axis=0)
    all_adapt_m = np.concatenate([niah_adapt_m, code_adapt_m, math_adapt_m, conv_adapt_m], axis=0)
    p_ad = np_rng.permutation(len(all_adapt_t))
    all_adapt_t, all_adapt_m = all_adapt_t[p_ad], all_adapt_m[p_ad]

    n_val_adapt = 400
    save_split_memmaps(out_dir, "adapt_val", all_adapt_t[:n_val_adapt], all_adapt_m[:n_val_adapt])
    save_split_memmaps(out_dir, "adapt_train", all_adapt_t[n_val_adapt:], all_adapt_m[n_val_adapt:])

    all_sft_t = np.concatenate([niah_sft_t, code_sft_t, math_sft_t, conv_sft_t], axis=0)
    all_sft_m = np.concatenate([niah_sft_m, code_sft_m, math_sft_m, conv_sft_m], axis=0)
    p_sf = np_rng.permutation(len(all_sft_t))
    all_sft_t, all_sft_m = all_sft_t[p_sf], all_sft_m[p_sf]

    n_val_sft = 400
    save_split_memmaps(out_dir, "sft_val", all_sft_t[:n_val_sft], all_sft_m[:n_val_sft])
    save_split_memmaps(out_dir, "sft_train", all_sft_t[n_val_sft:], all_sft_m[n_val_sft:])

    audit["total_adapt_train_tokens"] = int((len(all_adapt_t) - n_val_adapt) * SEQ_LEN)
    audit["total_sft_train_tokens"] = int((len(all_sft_t) - n_val_sft) * SEQ_LEN)
    audit["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    with open(os.path.join(out_dir, "audit_report_v2_1.json"), "w") as f:
        json.dump(audit, f, indent=2)

    print("\n" + "=" * 96, flush=True)
    print("VAULT v2.1 DATA HYGIENE & OPEN-VOCABULARY AUDIT COMPLETE!", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    main()
