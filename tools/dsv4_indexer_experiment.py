#!/usr/bin/env python3
"""Patch, run, collect, and plot a DeepSeek-V4 indexer threshold experiment.

This host-side tool intentionally uses only the Python standard library.  The
model container supplies torch/torch_npu; the host only sends OpenAI-compatible
requests, copies JSONL traces out of Docker, and renders SVG plots.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Running this file directly puts tools/ first on sys.path.  Remove it before
# urllib loads random -> bisect, which would otherwise resolve tools/bisect.
_SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _SCRIPT_DIR:
    sys.path.pop(0)

DEFAULT_TRACE_DIR = "/tmp/dsv4-indexer-trace"
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
CONTAINER_PATH_MARKER = "__DSV4_ATTENTION_DIR__="


def escape(value: str) -> str:
    """Escape text nodes without importing urllib through xml.sax."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run_command(args: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def source_files() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    attention_dir = repo_root / "vllm_ascend" / "attention"
    return attention_dir / "dsa_v1.py", attention_dir / "indexer_trace.py"


def container_attention_dir(container: str) -> str:
    code = (
        "import importlib.util; from pathlib import Path; "
        "spec = importlib.util.find_spec('vllm_ascend'); "
        "assert spec is not None and spec.submodule_search_locations; "
        f"print('{CONTAINER_PATH_MARKER}' + "
        "str(Path(next(iter(spec.submodule_search_locations))).resolve() / 'attention'))"
    )
    output = run_command(["docker", "exec", container, "python", "-c", code], capture=True)
    for line in output.splitlines():
        if line.startswith(CONTAINER_PATH_MARKER):
            return line.removeprefix(CONTAINER_PATH_MARKER).strip()
    raise RuntimeError(f"Could not locate vllm_ascend/attention in container {container}. Output: {output}")


def patch_container(container: str) -> None:
    local_dsa, local_trace = source_files()
    remote_dir = container_attention_dir(container)
    remote_dsa = f"{remote_dir}/dsa_v1.py"
    remote_trace = f"{remote_dir}/indexer_trace.py"
    backup_code = """
import pathlib, shutil, sys
for value in sys.argv[1:]:
    path = pathlib.Path(value)
    backup = path.with_name(path.name + '.before-indexer-trace')
    missing = path.with_name(path.name + '.missing-before-indexer-trace')
    if not backup.exists() and not missing.exists():
        if path.exists():
            shutil.copy2(path, backup)
        else:
            missing.touch()
""".strip()
    run_command(["docker", "exec", container, "python", "-c", backup_code, remote_dsa, remote_trace])
    run_command(["docker", "cp", str(local_dsa), f"{container}:{remote_dsa}"])
    run_command(["docker", "cp", str(local_trace), f"{container}:{remote_trace}"])

    digest_code = """
import hashlib, pathlib, sys
for value in sys.argv[1:]:
    print(hashlib.sha256(pathlib.Path(value).read_bytes()).hexdigest())
""".strip()
    remote_digests = run_command(
        ["docker", "exec", container, "python", "-c", digest_code, remote_dsa, remote_trace],
        capture=True,
    ).splitlines()
    local_digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (local_dsa, local_trace)]
    if remote_digests != local_digests:
        raise RuntimeError("Container patch verification failed: SHA-256 mismatch")
    print(f"Patched {remote_dsa}")
    print(f"Installed {remote_trace}")
    print("Patch verified. Start or restart vllm serve with the arguments shown by launch-args.")


def restore_container(container: str) -> None:
    remote_dir = container_attention_dir(container)
    restore_code = """
import pathlib, shutil, sys
for value in sys.argv[1:]:
    path = pathlib.Path(value)
    backup = path.with_name(path.name + '.before-indexer-trace')
    missing = path.with_name(path.name + '.missing-before-indexer-trace')
    if backup.exists():
        shutil.copy2(backup, path)
        backup.unlink()
    elif missing.exists() and path.exists():
        path.unlink()
    if missing.exists():
        missing.unlink()
""".strip()
    run_command(
        [
            "docker",
            "exec",
            container,
            "python",
            "-c",
            restore_code,
            f"{remote_dir}/dsa_v1.py",
            f"{remote_dir}/indexer_trace.py",
        ]
    )
    print("Restored container files. Restart vllm serve before using the service again.")


def launch_arguments(trace_dir: str, layers: list[int]) -> list[str]:
    trace_config: dict[str, Any] = {
        "enabled": True,
        "strict": True,
        "output_dir": trace_dir,
        "sample_every": 1,
    }
    if layers:
        trace_config["layers"] = layers
    additional_config = {
        "enable_dsa_cp": False,
        "multistream_overlap_shared_expert": False,
        "multistream_dsv4_dsa_overlap": False,
        "dsv4_indexer_trace": trace_config,
    }
    return [
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--no-enable-prefix-caching",
        "--hf-overrides",
        json.dumps({"use_index_cache": False}, separators=(",", ":")),
        "--additional-config",
        json.dumps(additional_config, separators=(",", ":")),
    ]


def print_launch_arguments(trace_dir: str, layers: list[int]) -> None:
    separator = " \\" + "\n  "
    print(separator.join(shlex.quote(arg) for arg in launch_arguments(trace_dir, layers)))
    print(
        "\nDo not add --speculative-config. If your existing command already has "
        "--additional-config or --hf-overrides, merge the JSON objects."
    )


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def wait_for_server(base_url: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return http_json("GET", f"{base_url.rstrip('/')}/v1/models", None, 10, api_key)
        except Exception as error:  # noqa: PERF203 - retry loop intentionally catches all request failures
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"vLLM service did not become ready within {timeout}s: {last_error}")


def synthetic_prompt(session_number: int, line_count: int = 260) -> str:
    topics = [
        ("orion", "cobalt", "Summarize the recurring project status and state the checkpoint color."),
        ("harbor", "amber", "Identify the repeated site name and its checkpoint color."),
        ("cedar", "violet", "Report the archive label and the checkpoint color in one sentence."),
    ]
    label, color, question = topics[session_number]
    lines = [
        f"Record {index:04d}: archive={label}; checkpoint={color}; sequence={index % 17}; "
        "this is stable background material for a long-context indexer measurement."
        for index in range(line_count)
    ]
    return "\n".join(lines + ["", question])


def load_prompts(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return [{"name": f"session-{index + 1}", "content": synthetic_prompt(index)} for index in range(3)]
    raw_prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_prompts, list) or len(raw_prompts) != 3:
        raise ValueError("Prompt JSON must be a list containing exactly three entries")
    prompts: list[dict[str, str]] = []
    for index, item in enumerate(raw_prompts):
        if isinstance(item, str):
            prompts.append({"name": f"session-{index + 1}", "content": item})
        elif isinstance(item, dict) and isinstance(item.get("content"), str):
            prompts.append(
                {
                    "name": str(item.get("name", f"session-{index + 1}")),
                    "content": item["content"],
                }
            )
        else:
            raise ValueError(f"Invalid prompt entry at index {index}")
    return prompts


def resolve_model(base_url: str, requested_model: str | None, models_response: dict[str, Any]) -> str:
    if requested_model:
        return requested_model
    models = models_response.get("data", [])
    if not models:
        raise RuntimeError(f"No model returned by {base_url}/v1/models; pass --model explicitly")
    return str(models[0]["id"])


def copy_trace_from_container(container: str, trace_dir: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    run_command(["docker", "cp", f"{container}:{trace_dir}/.", str(destination)])


def read_trace_records(trace_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(trace_dir.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    return records


def assign_sessions(records: list[dict[str, Any]], markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        timestamp = int(record["wall_time_ns"])
        for session_index, marker in enumerate(markers, 1):
            if marker["start_ns"] <= timestamp <= marker["end_ns"]:
                enriched = dict(record)
                enriched["session"] = marker["name"]
                enriched["session_index"] = session_index
                selected.append(enriched)
                break

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        grouped[(record["session_index"], record["layer"])].append(record)
    for group in grouped.values():
        group.sort(key=lambda record: (record["wall_time_ns"], record["context_len"]))
        for step, record in enumerate(group):
            record["session_step"] = step
    return sorted(selected, key=lambda record: (record["session_index"], record["wall_time_ns"]))


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "session",
        "session_index",
        "session_step",
        "layer",
        "context_len",
        "compressed_len",
        "topk",
        "valid_selected",
        "has_full_topk",
        "score_source",
        "cutoff",
        "top1",
        "mean_selected",
        "std_selected",
        "cutoff_over_top1",
        "wall_time_ns",
        "data_parallel_rank",
        "tensor_parallel_rank",
        "rank",
        "pid",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def layer_sort_key(layer_name: str) -> tuple[int, str]:
    match = LAYER_RE.search(layer_name)
    return (int(match.group(1)) if match else 1_000_000, layer_name)


def layer_label(layer_name: str) -> str:
    match = LAYER_RE.search(layer_name)
    return f"L{match.group(1)}" if match else layer_name


def color_for_index(index: int, total: int) -> str:
    red, green, blue = colorsys.hsv_to_rgb(index / max(total, 1), 0.72, 0.78)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def render_svg(records: list[dict[str, Any]], path: Path, metric: str, title: str) -> None:
    sessions = sorted({(record["session_index"], record["session"]) for record in records})
    layers = sorted({record["layer"] for record in records}, key=layer_sort_key)
    colors = {layer: color_for_index(index, len(layers)) for index, layer in enumerate(layers)}
    width = 1500
    panel_height = 360
    left, right, top, bottom = 85, 260, 55, 55
    plot_width = width - left - right
    height = max(1, len(sessions)) * panel_height
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" '
        f'font-family="sans-serif">{escape(title)}</text>',
    ]

    for panel_index, (session_index, session_name) in enumerate(sessions):
        panel_top = panel_index * panel_height
        x0, y0 = left, panel_top + top
        plot_height = panel_height - top - bottom
        session_records = [
            record
            for record in records
            if record["session_index"] == session_index
            and record.get("has_full_topk")
            and record.get(metric) is not None
            and math.isfinite(float(record[metric]))
        ]
        values = [float(record[metric]) for record in session_records]
        max_step = max((int(record["session_step"]) for record in session_records), default=1)
        if not values:
            elements.append(
                f'<text x="{x0}" y="{y0 + 30}" font-family="sans-serif" font-size="14">'
                f"{escape(session_name)}: no full top-k records</text>"
            )
            continue
        value_min, value_max = min(values), max(values)
        if math.isclose(value_min, value_max):
            padding = max(abs(value_min) * 0.05, 1e-6)
            value_min -= padding
            value_max += padding

        def x_coordinate(
            step: int,
            panel_x0: float = x0,
            panel_max_step: int = max_step,
        ) -> float:
            return panel_x0 + plot_width * step / max(panel_max_step, 1)

        def y_coordinate(
            value: float,
            panel_y0: float = y0,
            panel_plot_height: float = plot_height,
            panel_value_min: float = value_min,
            panel_value_max: float = value_max,
        ) -> float:
            return panel_y0 + panel_plot_height * (panel_value_max - value) / (panel_value_max - panel_value_min)

        elements.extend(
            [
                f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_height}" stroke="#333"/>',
                f'<line x1="{x0}" y1="{y0 + plot_height}" x2="{x0 + plot_width}" '
                f'y2="{y0 + plot_height}" stroke="#333"/>',
                f'<text x="{x0}" y="{panel_top + 48}" font-family="sans-serif" font-size="15">'
                f"{escape(session_name)}</text>",
                f'<text x="{x0 - 8}" y="{y0 + 5}" text-anchor="end" font-family="monospace" '
                f'font-size="11">{value_max:.5g}</text>',
                f'<text x="{x0 - 8}" y="{y0 + plot_height}" text-anchor="end" '
                f'font-family="monospace" font-size="11">{value_min:.5g}</text>',
                f'<text x="{x0 + plot_width / 2}" y="{y0 + plot_height + 38}" text-anchor="middle" '
                'font-family="sans-serif" font-size="12">decode step</text>',
            ]
        )
        for layer in layers:
            points = sorted(
                (
                    (int(record["session_step"]), float(record[metric]))
                    for record in session_records
                    if record["layer"] == layer
                ),
                key=lambda point: point[0],
            )
            if not points:
                continue
            coordinate_text = " ".join(f"{x_coordinate(step):.2f},{y_coordinate(value):.2f}" for step, value in points)
            elements.append(
                f'<polyline points="{coordinate_text}" fill="none" stroke="{colors[layer]}" '
                'stroke-width="1.4" opacity="0.86"/>'
            )

    legend_x = width - right + 25
    for index, layer in enumerate(layers):
        legend_y = 54 + (index % 22) * 14
        legend_column = index // 22
        x = legend_x + legend_column * 70
        elements.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 16}" y2="{legend_y}" stroke="{colors[layer]}"/>')
        elements.append(
            f'<text x="{x + 20}" y="{legend_y + 4}" font-family="monospace" font-size="10">'
            f"{escape(layer_label(layer))}</text>"
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("has_full_topk") and math.isfinite(float(record["cutoff"])):
            grouped[(record["session"], record["layer"])].append(record)
    rows = []
    for (session, layer), group in sorted(grouped.items(), key=lambda item: (item[0][0], layer_sort_key(item[0][1]))):
        group.sort(key=lambda record: record["session_step"])
        values = [float(record["cutoff"]) for record in group]
        rows.append(
            {
                "session": session,
                "layer": layer,
                "samples": len(values),
                "cutoff_min": min(values),
                "cutoff_max": max(values),
                "cutoff_mean": sum(values) / len(values),
                "first_to_last_change": values[-1] - values[0],
            }
        )
    return {
        "record_count": len(records),
        "full_topk_record_count": sum(bool(record.get("has_full_topk")) for record in records),
        "sessions": sorted({record["session"] for record in records}),
        "layers": sorted({record["layer"] for record in records}, key=layer_sort_key),
        "per_session_layer": rows,
    }


def analyze_trace(raw_trace_dir: Path, markers: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    records = assign_sessions(read_trace_records(raw_trace_dir), markers)
    if not records:
        raise RuntimeError(
            "No trace records matched the three requests. Check that the patched service was restarted "
            "with tracing enabled and inspect the vLLM worker log."
        )
    missing_sessions = {marker["name"] for marker in markers} - {record["session"] for record in records}
    if missing_sessions:
        raise RuntimeError(f"No trace records for sessions: {sorted(missing_sessions)}")
    write_csv(records, output_dir / "indexer_thresholds.csv")
    render_svg(records, output_dir / "cutoff.svg", "cutoff", "DeepSeek-V4 top-k cutoff by decode step")
    render_svg(
        records,
        output_dir / "normalized_cutoff.svg",
        "cutoff_over_top1",
        "DeepSeek-V4 normalized top-k cutoff by decode step",
    )
    summary = summarize(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir or f"dsv4-indexer-results-{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=False)
    models_response = wait_for_server(args.base_url, args.ready_timeout, args.api_key)
    model = resolve_model(args.base_url, args.model, models_response)
    prompts = load_prompts(Path(args.prompts_json) if args.prompts_json else None)
    markers: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []

    for prompt in prompts:
        print(f"Sending {prompt['name']} ({len(prompt['content'])} characters)...", flush=True)
        start_ns = time.time_ns()
        response = http_json(
            "POST",
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt["content"]}],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "stream": False,
            },
            args.request_timeout,
            args.api_key,
        )
        end_ns = time.time_ns()
        markers.append({"name": prompt["name"], "start_ns": start_ns, "end_ns": end_ns})
        responses.append({"name": prompt["name"], "response": response})
        print(f"Completed {prompt['name']}.", flush=True)

    (output_dir / "request_markers.json").write_text(
        json.dumps(markers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "responses.json").write_text(
        json.dumps(responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    time.sleep(1)
    raw_trace_dir = output_dir / "raw_trace"
    copy_trace_from_container(args.container, args.trace_dir, raw_trace_dir)
    records = analyze_trace(raw_trace_dir, markers, output_dir)
    print(f"Collected {len(records)} records.")
    print(f"CSV: {output_dir / 'indexer_thresholds.csv'}")
    print(f"Raw plot: {output_dir / 'cutoff.svg'}")
    print(f"Normalized plot: {output_dir / 'normalized_cutoff.svg'}")
    print(f"Summary: {output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    patch_parser = subparsers.add_parser("patch-container", help="Copy the two patched Python files into a container")
    patch_parser.add_argument("--container", required=True)

    restore_parser = subparsers.add_parser("restore-container", help="Restore the container's original Python files")
    restore_parser.add_argument("--container", required=True)

    launch_parser = subparsers.add_parser("launch-args", help="Print the required vllm serve arguments")
    launch_parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    launch_parser.add_argument("--layers", type=int, nargs="*", default=[])

    run_parser = subparsers.add_parser("run", help="Send three requests, collect traces, and make plots")
    run_parser.add_argument("--container", required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    run_parser.add_argument("--model")
    run_parser.add_argument("--api-key")
    run_parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    run_parser.add_argument("--prompts-json")
    run_parser.add_argument("--max-tokens", type=int, default=64)
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--ready-timeout", type=float, default=600)
    run_parser.add_argument("--request-timeout", type=float, default=3600)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "patch-container":
            patch_container(args.container)
        elif args.command == "restore-container":
            restore_container(args.container)
        elif args.command == "launch-args":
            print_launch_arguments(args.trace_dir, args.layers)
        elif args.command == "run":
            run_experiment(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
