"""The memory plan of a launch plan (``tilefish._plan.plan_arena``): offsets, alignment, and liveness reuse, checked against a simulated arena.

A step reads some buffers and writes one; steps run in order on one stream. ``plan_arena`` gives every buffer an offset in one arena. Without ``reuse`` every
buffer has a region of its own. With it two buffers share bytes only where their live ranges (the first to the last step that touches one, both inclusive)
do not intersect. The simulation runs random programs over the offsets and checks the only thing that matters: every read finds what its writer wrote.
"""

import random

import pytest

from tilefish import _plan
from tilefish._plan import ArenaItem, align_up, plan_arena


def program(rng, max_steps=16):
    """Random steps, each reading earlier results and writing a fresh buffer; some buffers (slots) are written by the host before the first step."""
    steps = []
    buffers = []  # (name, bytes, writer step or -1 for the host)
    for i in range(rng.randint(0, 3)):
        buffers.append((f"slot{i}", rng.choice([4, 8, 512]), -1))
    for s in range(rng.randint(1, max_steps)):
        readable = [b for b in buffers]
        reads = rng.sample(readable, min(len(readable), rng.randint(0, 3)))
        out = (f"b{s}", rng.choice([1, 100, 256, 257, 1000, 4096]), s)
        steps.append(([r[0] for r in reads], out[0]))
        buffers.append(out)
    return steps, buffers


def items_of(steps, buffers):
    first, last = {}, {}
    for s, (reads, out) in enumerate(steps):
        for name in [*reads, out]:
            first.setdefault(name, s)
            last[name] = s
    items = []
    for name, size, writer in buffers:
        if writer == -1:  # a slot lives for the whole plan, like the real ones
            items.append(ArenaItem(name, size, 0, len(steps) - 1, False))
        else:
            items.append(ArenaItem(name, size, first[name], last[name], True))
    return items


def simulate(steps, buffers, items, offsets, arena):
    """Run the steps over a byte array: a write stamps the region of its buffer, a read requires its region to still hold its own stamp."""
    memory = bytearray(arena)
    stamp = {name: (i % 250) + 1 for i, (name, _, _) in enumerate(buffers)}
    size = {name: s for name, s, _ in buffers}

    def fill(name):
        memory[offsets[name] : offsets[name] + size[name]] = bytes([stamp[name]]) * size[name]

    for name, _, writer in buffers:
        if writer == -1:
            fill(name)  # the host writes a slot before the plan runs
    for reads, out in steps:
        for r in reads:
            assert bytes(memory[offsets[r] : offsets[r] + size[r]]) == bytes([stamp[r]]) * size[r], f"{r} was overwritten before it was read"
        fill(out)


@pytest.mark.parametrize("seed", range(300))
@pytest.mark.parametrize("reuse", [False, True])
def test_every_read_finds_what_its_writer_wrote_whatever_the_program(seed, reuse):
    rng = random.Random(seed)
    steps, buffers = program(rng)
    items = items_of(steps, buffers)
    offsets, arena = plan_arena(items, reuse)
    assert all(offsets[i.name] % 256 == 0 for i in items)
    assert all(offsets[i.name] + align_up(i.bytes) <= arena for i in items)
    assert arena % 256 == 0
    simulate(steps, buffers, items, offsets, arena)
    if not reuse:  # a region of its own
        regions = sorted((offsets[i.name], offsets[i.name] + align_up(i.bytes)) for i in items)
        assert all(hi <= nxt for (_, hi), (nxt, _) in zip(regions, regions[1:]))
        assert arena == sum(align_up(i.bytes) for i in items)


@pytest.mark.parametrize("seed", range(300))
def test_reuse_never_needs_more_than_no_reuse_and_is_deterministic(seed):
    steps, buffers = program(random.Random(seed))
    items = items_of(steps, buffers)
    plain = plan_arena(items, False)
    shared = plan_arena(items, True)
    assert shared[1] <= plain[1]
    assert plan_arena(items, True) == shared and plan_arena(items, False) == plain


def test_the_simulation_can_fail_an_arena_that_ignores_liveness():
    """A control: with every buffer at offset 0 the simulation must catch the overwrite, or it checks nothing."""
    caught = 0
    for seed in range(100):
        steps, buffers = program(random.Random(seed))
        if len(steps) < 2:
            continue
        items = items_of(steps, buffers)
        offsets = {i.name: 0 for i in items}
        try:
            simulate(steps, buffers, items, offsets, max(align_up(i.bytes) for i in items))
        except AssertionError:
            caught += 1
    assert caught > 20


def test_a_step_that_reads_one_buffer_and_writes_another_has_both_live_so_they_never_share():
    # step 0 writes a; step 1 reads a and writes b; step 2 reads b and writes c. a and b are both live in step 1: no sharing. a and c may.
    items = [ArenaItem("a", 1000, 0, 1, True), ArenaItem("b", 1000, 1, 2, True), ArenaItem("c", 1000, 2, 2, True)]
    offsets, arena = plan_arena(items, True)
    assert offsets["a"] != offsets["b"] and offsets["b"] != offsets["c"]
    assert offsets["c"] == offsets["a"] and arena == 2 * 1024  # c reuses a's region, and only that
    assert plan_arena(items, False)[1] == 3 * 1024


def test_the_whole_plan_buffers_come_first_and_are_never_shared():
    items = [ArenaItem("a", 100, 0, 0, True), ArenaItem("slot", 4, 0, 5, False), ArenaItem("b", 100, 3, 3, True), ArenaItem("ws", 3000, 0, 5, False)]
    offsets, arena = plan_arena(items, True)
    assert offsets["slot"] == 0 and offsets["ws"] == 256  # in the order given
    assert offsets["a"] == offsets["b"] == 256 + 3072 and arena == 256 + 3072 + 256


def test_without_reuse_the_order_given_is_the_order_in_the_arena():
    items = [ArenaItem("x", 300, 0, 0, True), ArenaItem("y", 1, 1, 1, True), ArenaItem("z", 256, 2, 2, False)]
    offsets, arena = plan_arena(items, False)
    assert offsets == {"x": 0, "y": 512, "z": 768} and arena == 1024


def test_an_empty_plan_has_an_empty_arena():
    assert plan_arena([], False) == ({}, 0) and plan_arena([], True) == ({}, 0)


def test_align_up():
    assert [align_up(n) for n in (0, 1, 255, 256, 257, 512)] == [0, 256, 256, 256, 512, 512] and _plan.ARENA_ALIGNMENT == 256
