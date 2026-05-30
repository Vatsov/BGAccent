from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from bgaccent.accentor import Accentor
from bgaccent.report import AccentResult

app = typer.Typer(add_completion=False)

SUPPORTED_MODES = ("preserve", "replace-safe")
SUPPORTED_FORMATS = ("text", "ssml")


def _finalize_text(text: str, output_format: str) -> str:
    if output_format == "ssml":
        from bgaccent.ssml import format_ssml

        return format_ssml(text)
    return text


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
    check: Annotated[
        bool,
        typer.Option("--check", help="Dry-run: output stats, no text"),
    ] = False,
    fail_on_oov: Annotated[
        bool,
        typer.Option("--fail-on-oov", help="Exit 1 if OOV words found (with --check)"),
    ] = False,
    fail_on_homographs: Annotated[
        bool,
        typer.Option("--fail-on-homographs", help="Exit 1 if homographs flagged"),
    ] = False,
    max_oov_rate: Annotated[
        float | None,
        typer.Option("--max-oov-rate", help="Exit 1 if OOV rate exceeds threshold"),
    ] = None,
    diff: Annotated[
        bool,
        typer.Option("--diff", help="Output per-occurrence changes"),
    ] = False,
    unknown_only: Annotated[
        bool,
        typer.Option("--unknown-only", help="Output OOV words only, one per line"),
    ] = False,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="Output directory for batch processing"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or ssml"),
    ] = "text",
) -> None:
    if mode not in SUPPORTED_MODES:
        typer.echo(f"Error: --mode must be one of {SUPPORTED_MODES}", err=True)
        raise typer.Exit(code=2)

    if output_format not in SUPPORTED_FORMATS:
        typer.echo(f"Error: --format must be one of {SUPPORTED_FORMATS}", err=True)
        raise typer.Exit(code=2)

    trie_path = trie
    if trie_path is None:
        from bgaccent.data import get_trie_path

        try:
            trie_path = get_trie_path()
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if input_arg is not None and Path(input_arg).is_dir():
        _handle_batch(
            input_dir=Path(input_arg),
            out_dir=out_dir,
            trie_path=trie_path,
            mark_monosyllables=mark_monosyllables,
            custom_dicts=list(custom) if custom else None,
            mode=mode,
            report_path=report,
            check=check,
            fail_on_oov=fail_on_oov,
            fail_on_homographs=fail_on_homographs,
            max_oov_rate=max_oov_rate,
            quiet=quiet,
            output_format=output_format,
        )
        return

    try:
        text = _read_input(input_arg)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except UnicodeDecodeError as exc:
        typer.echo(
            "Error: input file is not valid UTF-8. "
            "Try converting with: iconv -f <encoding> -t utf-8 <file>",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    acc = Accentor(
        trie_path=trie_path,
        mark_monosyllables=mark_monosyllables,
        custom_dicts=list(custom) if custom else None,
        mode=mode,  # type: ignore[arg-type]
    )

    result = acc.accent_with_report(text)

    if report:
        report.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if check:
        _handle_check(result, fail_on_oov, fail_on_homographs, max_oov_rate, quiet)
        return

    if diff:
        _handle_diff(result)
        return

    if unknown_only:
        _handle_unknown_only(result)
        return

    final_text = _finalize_text(result.text, output_format)

    if output:
        output.write_text(final_text, encoding="utf-8")
        _write_log(output, result, quiet)
    else:
        typer.echo(final_text, nl=False)


def _handle_batch(
    input_dir: Path,
    out_dir: Path | None,
    trie_path: Path,
    mark_monosyllables: bool,
    custom_dicts: list[Path] | None,
    mode: str,
    report_path: Path | None,
    check: bool,
    fail_on_oov: bool,
    fail_on_homographs: bool,
    max_oov_rate: float | None,
    quiet: bool,
    output_format: str,
) -> None:
    from bgaccent.report import AccentResult, AccentStats

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        if not quiet:
            typer.echo("Warning: no .txt files found in input directory", err=True)
        raise typer.Exit(code=0)

    acc = Accentor(
        trie_path=trie_path,
        mark_monosyllables=mark_monosyllables,
        custom_dicts=custom_dicts,
        mode=mode,  # type: ignore[arg-type]
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    report_dir: Path | None = None
    if report_path is not None:
        report_dir = report_path
        report_dir.mkdir(parents=True, exist_ok=True)

    total_stats = AccentStats()
    all_oov: list[str] = []
    oov_seen: set[str] = set()
    all_homographs: list[str] = []
    hom_seen: set[str] = set()
    per_file: list[dict[str, object]] = []

    for txt_file in txt_files:
        if not quiet:
            typer.echo(f"Processing {txt_file.name}...", err=True)

        text = txt_file.read_text(encoding="utf-8-sig")
        result = acc.accent_with_report(text)

        if out_dir is not None and not check:
            final_text = _finalize_text(result.text, output_format)
            (out_dir / txt_file.name).write_text(final_text, encoding="utf-8")
            _write_log(out_dir / txt_file.name, result, quiet)

        if report_dir is not None:
            report_file = report_dir / txt_file.with_suffix(".json").name
            report_file.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        total_stats += result.stats

        for w in result.oov_words:
            if w not in oov_seen:
                oov_seen.add(w)
                all_oov.append(w)
        for h in result.homographs:
            if h not in hom_seen:
                hom_seen.add(h)
                all_homographs.append(h)

        per_file.append({"file": txt_file.name, "stats": result.to_dict()["stats"]})

    if report_dir is not None:
        summary = {
            "total": {
                "total_tokens": total_stats.total_tokens,
                "accented_tokens": total_stats.accented_tokens,
                "oov_multisyllabic": total_stats.oov_multisyllabic,
                "homographs_flagged": total_stats.homographs_flagged,
            },
            "oov_words": all_oov,
            "homographs": all_homographs,
            "per_file": per_file,
        }
        (report_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if check:
        agg = AccentResult(text="", stats=total_stats, oov_words=all_oov, homographs=all_homographs)
        _handle_check(agg, fail_on_oov, fail_on_homographs, max_oov_rate, quiet)
        return


def _handle_check(
    result: AccentResult,
    fail_on_oov: bool,
    fail_on_homographs: bool,
    max_oov_rate: float | None,
    quiet: bool,
) -> None:
    stats = result.stats
    if not quiet:
        typer.echo(
            f"total_tokens: {stats.total_tokens}\n"
            f"accented_tokens: {stats.accented_tokens}\n"
            f"oov_multisyllabic: {stats.oov_multisyllabic}\n"
            f"homographs_flagged: {stats.homographs_flagged}",
            err=True,
        )

    if fail_on_oov and stats.oov_multisyllabic > 0:
        raise typer.Exit(code=1)
    if fail_on_homographs and stats.homographs_flagged > 0:
        raise typer.Exit(code=1)
    if max_oov_rate is not None and stats.total_tokens > 0:
        rate = stats.oov_multisyllabic / stats.total_tokens
        if rate > max_oov_rate:
            raise typer.Exit(code=1)


def _handle_diff(result: AccentResult) -> None:
    lines: list[str] = []
    for detail in result.details:
        if detail.get("status") in ("accented", "homograph_flagged", "custom_override"):
            word = detail["word"]
            accented = detail.get("accented", word)
            if word != accented:
                line = detail.get("line", 0)
                col = detail.get("column", 0)
                lines.append(f"L{line}:{col} {word} -> {accented}")
    if lines:
        typer.echo("\n".join(lines))


def _handle_unknown_only(result: AccentResult) -> None:
    if result.oov_words:
        typer.echo("\n".join(result.oov_words))


def _write_log(output: Path, result: AccentResult, quiet: bool) -> None:
    log_path = output.with_suffix(".log")
    stats = result.stats
    log_lines = [
        f"total_tokens: {stats.total_tokens}",
        f"accented_tokens: {stats.accented_tokens}",
        f"oov_multisyllabic: {stats.oov_multisyllabic}",
        f"skipped_monosyllabic: {stats.skipped_monosyllabic}",
        f"homographs_flagged: {stats.homographs_flagged}",
        f"custom_overrides: {stats.custom_overrides}",
    ]
    if result.oov_words:
        log_lines.append(f"oov_words: {', '.join(result.oov_words)}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _read_input(input_arg: str | None) -> str:
    if input_arg is None:
        return sys.stdin.read()

    if input_arg == "-":
        return sys.stdin.read()

    path = Path(input_arg)
    if path.is_file():
        return path.read_text(encoding="utf-8-sig")

    if path.suffix and not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    return input_arg
