"""MkDocs hooks for the documentation site (mkdocs.yml).

The pages are written for GitHub, and the site shows them as they are, with three changes made here before MkDocs validates the links (so a
strict build still refuses a link to a page or an anchor that does not exist):

* a relative link that leaves the site (the code, the CHANGELOG, the READMEs, a directory) goes to the file on GitHub (``blob/main/...``,
  ``tree/main/...`` for a directory);
* a link from a page of one language to a page of the other (``docs/en/`` and ``docs/zh-CN/`` are built one at a time) goes to that page on
  the site;
* the line that switches language under the title is dropped: the site has its own language switcher, which keeps the page.

And two differences between GitHub's Markdown and Python-Markdown are smoothed over: a ``\\|`` in a code span of a table row (GitHub needs the
backslash, Python-Markdown shows it) loses its backslash, and a list item right after a paragraph (where GitHub starts a list and
Python-Markdown reads more of the paragraph) gets a blank line before it. Nested lists by two spaces are the ``mdx_truly_sane_lists``
extension's (mkdocs.yml).

Links between pages of one language and to anchors are left to MkDocs. Fenced code blocks are not touched, nor inline code but for the ``\\|``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

GITHUB = "https://github.com/Kilerd/tilefish"
BRANCH = "main"
LANGUAGES = ("en", "zh-CN")  # docs/<locale>/, as in mkdocs.yml; the first is the default, at the root of the site

_FENCE = re.compile(r"^\s*(```+|~~~+)")
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)")
_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?((?:\s+\"[^\"]*\")?)\s*\)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _is_switch_line(line: str) -> bool:
    return ("**English**" in line and "[简体中文](" in line) or ("[English](" in line and "**简体中文**" in line)


def _site_page_url(site_url: str, docs_dir: Path, page: Path) -> str:
    """The URL of a page on the site (directory URLs): docs/zh-CN/guide/plans.md -> <site>/zh-CN/guide/plans/."""
    lang, *rest = page.relative_to(docs_dir).parts
    path = "/".join(rest)[: -len(".md")]
    path = "" if path in ("README", "index") else path.removesuffix("/README").removesuffix("/index") + "/"
    return site_url.rstrip("/") + "/" + ("" if lang == LANGUAGES[0] else lang + "/") + path


def _new_target(target: str, page_src: Path, docs_dir: Path, repo_root: Path, site_url: str) -> str:
    """Where a relative link of a page goes on the site, or the link as it is."""
    if _SCHEME.match(target) or target.startswith(("#", "/")):
        return target
    path, sep, anchor = target.partition("#")
    if not path:
        return target
    resolved = Path(os.path.normpath(page_src.parent / path))
    inside = resolved == docs_dir or docs_dir in resolved.parents
    if inside and resolved.suffix == ".md" and resolved.is_file():
        page_lang = page_src.relative_to(docs_dir).parts[0]
        target_lang = resolved.relative_to(docs_dir).parts[0]
        if target_lang != page_lang and target_lang in LANGUAGES and site_url:
            return _site_page_url(site_url, docs_dir, resolved) + (sep + anchor if sep else "")
        return target  # a page of this language: MkDocs resolves it (and refuses it if it is not one)
    if not resolved.exists() or not (resolved == repo_root or repo_root in resolved.parents):
        return target  # not a file of the repository: MkDocs reports it
    kind = "tree" if resolved.is_dir() else "blob"
    rel = resolved.relative_to(repo_root).as_posix()
    return f"{GITHUB}/{kind}/{BRANCH}/{rel}" + (sep + anchor if sep else "")


def _rewrite_prose(text: str, *where) -> str:
    def links(s: str) -> str:
        return _LINK.sub(lambda m: "](" + _new_target(m.group(1), *where) + m.group(2) + ")", s)

    out, pos = [], 0
    for m in _CODE_SPAN.finditer(text):
        out += [links(text[pos : m.start()]), m.group(0)]
        pos = m.end()
    out.append(links(text[pos:]))
    return "".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:
    docs_dir = Path(config["docs_dir"]).resolve()
    repo_root = Path(config["config_file_path"]).resolve().parent
    page_src = Path(page.file.abs_src_path).resolve()
    where = (page_src, docs_dir, repo_root, config.get("site_url") or "")
    lines, out, para, fence = markdown.split("\n"), [], [], None

    # the switch line: the first non-empty line after the title
    for i, line in enumerate(lines):
        if line.startswith("# "):
            j = next((k for k in range(i + 1, len(lines)) if lines[k].strip()), None)
            if j is not None and _is_switch_line(lines[j]):
                del lines[j]
            break

    def flush():
        if para:
            text = _rewrite_prose("\n".join(para), *where)
            out.extend(_TABLE_ROW_PIPES(row) if row.lstrip().startswith("|") else row for row in text.split("\n"))
            para.clear()

    for line in lines:
        m = _FENCE.match(line)
        if fence is None and m:
            flush()
            fence = m.group(1)
            out.append(line)
            continue
        if fence is not None:
            if line.strip() == fence:
                fence = None
            out.append(line)
            continue
        if not line.strip():
            flush()
            out.append(line)
            continue
        # a list item right after a paragraph that is not a list item, or right after a code block: GitHub starts the list (or the next item)
        # there, Python-Markdown wants a blank line first
        if _LIST_ITEM.match(line):
            after_paragraph = para and not _LIST_ITEM.match(para[0]) and not para[0].lstrip().startswith("|")
            after_fence = not para and out and _FENCE.match(out[-1]) is not None
            if after_paragraph or after_fence:
                flush()
                out.append("")
        para.append(line)
    flush()
    return "\n".join(out)


_LIST_ITEM = re.compile(r"^\s*(?:[*+-]|\d+\.)\s")


def _TABLE_ROW_PIPES(row: str) -> str:
    """A table row with ``\\|`` in a code span as Python-Markdown wants it: ``|`` (it does not split a cell inside a code span)."""
    return _CODE_SPAN.sub(lambda m: m.group(0).replace("\\|", "|"), row)


try:  # jieba (the search plugin's Chinese segmenter) tells its progress on the root logger's level; it is not a warning of the build
    import logging

    import jieba

    jieba.setLogLevel(logging.WARNING)
except ImportError:
    pass
