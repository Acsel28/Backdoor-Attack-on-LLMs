import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Finding:
    severity: str  # LOW | MEDIUM | HIGH
    kind: str
    message: str


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_chat_template(model_dir: Path) -> Tuple[Optional[str], List[str]]:
    """Return (template, sources) where sources lists which files contained it."""
    sources: List[str] = []

    tokenizer_config = _read_json(model_dir / "tokenizer_config.json") or {}
    template = tokenizer_config.get("chat_template")
    if isinstance(template, str) and template.strip():
        sources.append("tokenizer_config.json:chat_template")

    # Some repos store templates in tokenizer.json (or nested).
    tokenizer_json = _read_json(model_dir / "tokenizer.json") or {}
    template2 = None

    # Common locations (best-effort, schema varies by tokenizer).
    if isinstance(tokenizer_json.get("chat_template"), str):
        template2 = tokenizer_json.get("chat_template")
    else:
        added_tokens = tokenizer_json.get("added_tokens")
        if isinstance(added_tokens, list):
            # Not a template, but sometimes templates or role markers appear here.
            pass

    if isinstance(template2, str) and template2.strip():
        if template is None:
            template = template2
        sources.append("tokenizer.json:chat_template")

    return template, sources


def _regex_find(patterns: Iterable[str], text: str) -> List[str]:
    hits: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(pattern)
    return hits


def analyze_template(template: str) -> Tuple[int, List[Finding]]:
    """Returns (risk_score, findings). Risk score is a heuristic for triage."""
    findings: List[Finding] = []
    risk = 0

    # Jinja control flow / executable logic.
    jinja_control = [
        r"\{%-?\s*if\b",
        r"\{%-?\s*elif\b",
        r"\{%-?\s*else\b",
        r"\{%-?\s*endif\b",
        r"\{%-?\s*for\b",
        r"\{%-?\s*endfor\b",
        r"\{%-?\s*set\b",
        r"\{%-?\s*macro\b",
        r"\{%-?\s*endmacro\b",
        r"\{%-?\s*include\b",
        r"\{%-?\s*import\b",
        r"\{%-?\s*from\b",
    ]
    hits = _regex_find(jinja_control, template)
    if hits:
        risk += 3
        findings.append(
            Finding(
                severity="MEDIUM",
                kind="jinja_control_flow",
                message=f"Template contains Jinja-like control flow ({len(hits)} pattern hits).",
            )
        )

    # Access to message content / roles (used benignly, but also for trigger checks).
    role_refs = [
        r"messages\[",
        r"message\.",
        r"role\b",
        r"content\b",
    ]
    hits = _regex_find(role_refs, template)
    if hits:
        risk += 1
        findings.append(
            Finding(
                severity="LOW",
                kind="message_role_refs",
                message="Template references message fields/roles (common; also enables conditional triggers).",
            )
        )

    # Signs of hidden system-instruction injection or role-marker construction.
    injection_markers = [
        r"system\b",
        r"<\|system\|>",
        r"\[\s*INST\s*\]",
        r"<<\s*SYS\s*>>",
        r"assistant\b",
        r"user\b",
        r"tool\b",
    ]
    hits = _regex_find(injection_markers, template)
    if hits:
        risk += 2
        findings.append(
            Finding(
                severity="LOW",
                kind="role_markers",
                message="Template contains role/system markers (usually expected, but relevant for template-backdoor threats).",
            )
        )

    # Heuristic: look for string matching checks inside conditionals.
    # e.g., `{% if "trigger" in message.content %}`
    trigger_like = [
        r"\bin\s+message\.content\b",
        r"\bin\s+messages\[",
        r"contains\(",
        r"startswith\(",
        r"endswith\(",
        r"replace\(",
        r"lower\(",
        r"upper\(",
        r"regex",
    ]
    hits = _regex_find(trigger_like, template)
    if hits and _regex_find([r"\{%-?\s*if\b"], template):
        risk += 4
        findings.append(
            Finding(
                severity="HIGH",
                kind="conditional_string_matching",
                message="Template appears to do conditional string matching/transforms (potential trigger check).",
            )
        )

    # Very long templates are harder to review and can hide logic.
    if len(template) > 8000:
        risk += 2
        findings.append(
            Finding(
                severity="MEDIUM",
                kind="very_long_template",
                message=f"Template is very long ({len(template)} chars); manual review recommended.",
            )
        )

    return risk, findings


def audit_model_dir(model_dir: Path) -> Dict[str, Any]:
    template, sources = _extract_chat_template(model_dir)

    result: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "has_chat_template": template is not None,
        "template_sources": sources,
        "risk_score": 0,
        "findings": [],
    }

    if template is None:
        return result

    risk, findings = analyze_template(template)
    result["risk_score"] = risk
    result["findings"] = [finding.__dict__ for finding in findings]

    # Provide a short excerpt to help triage without dumping the whole template.
    excerpt_len = 600
    cleaned = template.strip().replace("\r\n", "\n")
    result["template_excerpt"] = cleaned[:excerpt_len] + ("\n...[truncated]" if len(cleaned) > excerpt_len else "")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Hugging Face chat templates/tokenizer artifacts for template-backdoor risk indicators."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to a local HF model directory (must contain tokenizer_config.json and/or tokenizer.json).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists() or not model_dir.is_dir():
        raise SystemExit(f"Not a directory: {model_dir}")

    report = audit_model_dir(model_dir)

    # Human-readable output
    print(f"Model dir: {report['model_dir']}")
    print(f"Has chat_template: {report['has_chat_template']}")
    if report["has_chat_template"]:
        print(f"Template sources: {', '.join(report['template_sources']) or '(unknown)'}")
        print(f"Risk score: {report['risk_score']}")
        if report["findings"]:
            print("Findings:")
            for f in report["findings"]:
                print(f"- [{f['severity']}] {f['kind']}: {f['message']}")
        else:
            print("Findings: (none)")
        if "template_excerpt" in report:
            print("\nTemplate excerpt:\n")
            print(report["template_excerpt"])

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON report: {out_path}")


if __name__ == "__main__":
    main()
