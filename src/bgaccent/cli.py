from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from bgaccent.accentor import Accentor

app = typer.Typer(add_completion=False)

SUPPORTED_MODES = ("preserve", "replace-safe")


def version_callback(value: bool) -> None:
    if value:
        version = importlib.metadata.version("bgaccent")
        typer.echo(f"bgaccent {version}")
        raise typer.Exit()


@app.command()
def main(
    input_arg: Annotated[
        str | None,
        typer.Argument(help="Input file path, '-' for stdin, or raw text"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output file path (default: stdout)"),
    ] = None,
    custom: Annotated[
        list[Path] | None,
        typer.Option("--custom", help="Custom dictionary TSV (repeatable)"),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Accent mode: preserve or replace-safe"),
    ] = "preserve",
    mark_monosyllables: Annotated[
        bool,
        typer.Option("--mark-monosyllables", help="Place accent on monosyllabic words"),
    ] = False,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write JSON report to file"),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-error stderr output"),
    ] = False,
    trie: Annotated[
        Path | None,
        typer.Option("--trie", help="Path to .marisa trie (default: bundled)"),
    ] = None,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    if mode not in SUPPORTED_MODES:
        typer.echo(f"Error: --mode must be one of {SUPPORTED_MODES}", err=True)
        raise typer.Exit(code=2)

    trie_path = trie
    if trie_path is None:
        from bgaccent.data import get_trie_path

        trie_path = get_trie_path()

    acc = Accentor(
        trie_path=trie_path,
        mark_monosyllables=mark_monosyllables,
        custom_dicts=list(custom) if custom else None,
        mode=mode,  # type: ignore[arg-type]
    )

    text = _read_input(input_arg)
    result = acc.accent_with_report(text)

    if output:
        output.write_text(result.text, encoding="utf-8")
    else:
        typer.echo(result.text, nl=False)

    if report:
        report.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _read_input(input_arg: str | None) -> str:
    if input_arg is None:
        return sys.stdin.read()

    if input_arg == "-":
        return sys.stdin.read()

    path = Path(input_arg)
    if path.is_file():
        return path.read_text(encoding="utf-8-sig")

    return input_arg
