"""The Simplified Chinese pages are mirrors of the English ones: ``README.md`` -> ``README.zh-CN.md``, ``python/README.md`` ->
``python/README.zh-CN.md``, ``crates/tilefish/README.md`` -> ``crates/tilefish/README.zh-CN.md``, ``docs/X.md`` -> ``docs/zh-CN/X.md``.

A mirror has the headings of its English page (the same levels in the same order), the same fenced code blocks byte for byte and in the
same order, and both pages have the language switch line right after the H1. Every English page has a mirror.
"""

import pathlib
import re

import pytest
from conftest import REPO_ROOT


def english_pages() -> list[pathlib.Path]:
    """The English pages that have a Chinese mirror: the three READMEs and every page under ``docs/`` but ``docs/zh-CN/``."""
    docs = REPO_ROOT / "docs"
    if not docs.is_dir():
        pytest.skip("docs/ is not available (not running from a source checkout)")
    pages = [REPO_ROOT / "README.md", REPO_ROOT / "python" / "README.md", REPO_ROOT / "crates" / "tilefish" / "README.md"]
    pages += sorted(p for p in docs.rglob("*.md") if "zh-CN" not in p.relative_to(docs).parts)
    return [p for p in pages if p.exists()]


def zh_path(en: pathlib.Path) -> pathlib.Path:
    rel = en.relative_to(REPO_ROOT)
    if rel.parts[0] == "docs":
        return REPO_ROOT / "docs" / "zh-CN" / pathlib.Path(*rel.parts[1:])
    return en.with_name(en.stem + ".zh-CN.md")


def parse(text: str) -> tuple[list[int], list[str]]:
    """The levels of the headings outside code blocks, and the fenced code blocks (fences included)."""
    heads, blocks, cur, fence = [], [], [], None
    for line in text.split("\n"):
        m = re.match(r"^(\s*)(```+|~~~+)(.*)$", line)
        if fence is None and m:
            fence, cur = m.group(2), [line]
            continue
        if fence is not None:
            cur.append(line)
            if line.strip() == fence:
                blocks.append("\n".join(cur))
                fence = None
            continue
        h = re.match(r"^(#{1,6})\s", line)
        if h:
            heads.append(len(h.group(1)))
    return heads, blocks


def has_switch_line(text: str, zh: bool) -> bool:
    """The first non-empty line after the H1 is ``**English** | [简体中文](...)`` (English) or ``[English](...) | **简体中文**`` (Chinese)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            nxt = next((x for x in lines[i + 1 :] if x.strip()), "")
            return ("**简体中文**" in nxt and "[English](" in nxt) if zh else ("**English**" in nxt and "[简体中文](" in nxt)
    return False


def test_every_zh_cn_mirror_matches_its_english_page():
    problems = []
    for en in english_pages():
        zh = zh_path(en)
        if not zh.exists():
            continue  # the completeness test reports it
        name, zh_name = en.relative_to(REPO_ROOT), zh.relative_to(REPO_ROOT)
        en_text, zh_text = en.read_text(encoding="utf-8"), zh.read_text(encoding="utf-8")
        (en_heads, en_blocks), (zh_heads, zh_blocks) = parse(en_text), parse(zh_text)
        if en_heads != zh_heads:
            problems.append(f"{zh_name}: heading levels {zh_heads} differ from those of {name}, {en_heads}")
        if len(en_blocks) != len(zh_blocks):
            problems.append(f"{zh_name}: {len(zh_blocks)} code blocks, {name} has {len(en_blocks)}")
        else:
            problems += [f"{zh_name}: code block {i + 1} differs from {name}'s" for i, (a, b) in enumerate(zip(en_blocks, zh_blocks)) if a != b]
        if not has_switch_line(en_text, zh=False):
            problems.append(f"{name}: no '**English** | [简体中文](...)' line after the H1")
        if not has_switch_line(zh_text, zh=True):
            problems.append(f"{zh_name}: no '[English](...) | **简体中文**' line after the H1")
    assert not problems, "\n".join(problems)


def test_completeness_every_english_page_has_a_zh_cn_mirror():
    missing = [f"{zh_path(en).relative_to(REPO_ROOT)} (of {en.relative_to(REPO_ROOT)})" for en in english_pages() if not zh_path(en).exists()]
    assert not missing, "no Chinese mirror:\n" + "\n".join(missing)
