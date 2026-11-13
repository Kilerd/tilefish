"""``docs/plan-schema-1.md`` says what the code does: its example is a valid plan that the code makes, its test vectors and its numbers are the ones computed here.

A specification for another implementation (the Rust lane) is only as good as its examples, so the examples are run.
"""

import json
import re

import plan_model as pm
import pytest
from conftest import REPO_ROOT

import tilefish
from tilefish import _plan

DOC = REPO_ROOT / "docs" / "plan-schema-1.md"


@pytest.fixture(scope="module")
def text():
    if not DOC.exists():
        pytest.skip("docs/plan-schema-1.md is not available (not running from a source checkout)")
    return DOC.read_text(encoding="utf-8")


def test_the_example_is_a_valid_plan_with_the_right_digest_and_is_what_the_code_makes(text):
    block = text.split("<!-- plan-example -->\n```json\n", 1)[1].split("\n```", 1)[0]
    doc = json.loads(block)
    _plan.validate_doc(doc)
    assert _plan.plan_digest(doc) == doc["plan_digest"]
    made = tilefish.plan(pm.accumulate, buckets={"Q": [16, 32]}, reuse=True).with_manifest("sm_103", "ab" * 32).doc
    assert {k: v for k, v in doc.items() if k != "tilefish"} == {k: v for k, v in made.items() if k != "tilefish"}


def test_the_digest_vectors_are_the_ones_the_code_computes(text):
    assert _plan.plan_digest({}) == "345611870fcf5832cd7533b55c9bcb061b78c17c4f53635c9bcf8f7cffca6893"
    assert "345611870fcf5832cd7533b55c9bcb061b78c17c4f53635c9bcf8f7cffca6893" in text
    tiny = {"schema": "tilefish.plan.v1", "tilefish": "x", "graph": "g", "plan_digest": "ignored"}
    assert _plan.plan_digest(tiny) == "f59e1096fcfb737bcb745fc0a11258a25a1e2eb2fd3039b8958c777d4dadc71e"
    assert "f59e1096fcfb737bcb745fc0a11258a25a1e2eb2fd3039b8958c777d4dadc71e" in text


def test_the_steps_with_views_and_the_arena_numbers_of_the_block_are_the_ones_the_plan_has(text):
    p = tilefish.plan(pm.block)
    shown = text.split("The steps of a plan with views", 1)[1].split("```json\n", 1)[1].split("\n```", 1)[0]
    attention = next(s for s in p.steps(M=64) if s["entry"] == "attention")
    assert json.loads(shown) == attention
    reused = tilefish.plan(pm.block, reuse=True)
    workspace = next(b for b in p.buffers if b["role"] == "workspace")["bytes"]
    figures = [p.arena_bytes, workspace, p.arena_bytes - workspace - 256, reused.arena_bytes, reused.arena_bytes - workspace - 256]
    assert figures == [9634048, 8519680, 1114112, 9240832, 720896]
    spaced = lambda n: f"{n:,}".replace(",", " ")  # noqa: E731
    for n in figures:
        assert spaced(n) in text, n


def test_every_key_and_role_the_document_names_is_the_code_s(text):
    for key in _plan.TOP_KEYS + _plan.BUFFER_KEYS + _plan.STEP_KEYS + list(_plan.ROLES):
        assert f"`{key}`" in text, key
    assert re.search(r"`\"tilefish.plan.v1\"`", text) and _plan.PLAN_SCHEMA == "tilefish.plan.v1"
    assert "b\"tilefish.plan-digest.v1\\0\"" in text and _plan.DIGEST_TAG == b"tilefish.plan-digest.v1\0"
