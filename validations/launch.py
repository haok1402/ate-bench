import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

VALIDATIONS = Path("validations")
TEMPLATE = Path(VALIDATIONS, "prompt.md")
SNAPSHOTS = Path(os.environ.get("SNAPSHOTS", "snapshots"))

LAYOUT_PATTERN = re.compile(r"challenges/(?P<suite>[^/]+)/(?P<task>[^/]+)/(?P<run>[^/]+)/patches$")

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def locate(patches: Path):
    """
    Derive the validation, the task, and the run from a patches directory, which the
    snapshot layout spells out: <snapshots>/challenges/<suite>/<task>/<run>/patches.

    Reading the task off the path is what keeps a snapshot from ever being judged against
    another task's rules, and it fixes where the results go: alongside the challenge traces
    under <snapshots>/validations/<suite>/<task>/<run>/, paired to the run by its id.
    """
    match = LAYOUT_PATTERN.search(patches.as_posix())
    if match is None:
        raise SystemExit(f"cannot locate a validation for {patches.as_posix()}: expected .../challenges/<suite>/<task>/<run>/patches")
    task = Path(match["suite"], match["task"])
    validation = Path(VALIDATIONS, task)
    if not Path(validation, "intro.md").is_file():
        raise SystemExit(f"no validation for {task.as_posix()}: {Path(validation, 'intro.md').as_posix()} is missing")
    return validation, task, match["run"]


def verdict_of(output: str):
    """
    The verdict the judge opened with, or None if it did not answer in the asked-for form.
    """
    for line in output.strip().splitlines():
        if line.strip():
            return line.strip() if line.strip() in ("PASS", "FAIL") else None
    return None


class Judge:
    """
    Judge one snapshot against one rule of one validation, in its own agent session.

    Each rule gets a separate session so no rule's evidence can leak into another's
    result, and each session's result is written on its own as soon as it lands.
    """

    def __init__(self, patches: Path, agent: str, model: str, rule: str):
        self.patches, self.agent, self.model, self.rule = patches, agent, model, rule
        self.validation, self.task, self.run = locate(patches)
        self.result = Path(SNAPSHOTS, VALIDATIONS, self.task, self.run, f"{rule}.md")

    def compose(self):
        """
        Compose the prompt from the shared template, the task's intro, and the one rule under
        test. The rule carries no identity into the prompt: which rule ran is the launcher's
        bookkeeping, held by the rule's filename, so the judge only ever sees one rule to
        weigh and cannot report on any other.
        """
        intro = Path(self.validation, "intro.md").read_text().strip()
        rule = Path(self.validation, f"{self.rule}.md").read_text().strip()
        return TEMPLATE.read_text().format(
            task=self.task.as_posix(),
            intro=intro,
            rule=rule,
            patches=self.patches.absolute().as_posix(),
        )

    def build_claude_command(self, prompt: str):
        if shutil.which("claude") is None:
            raise SystemExit("missing the required tool: claude")
        args = ["claude", "--print"]
        args.extend(["--model", self.model, "--effort", "xhigh"])
        args.extend(["--disallowedTools", "Edit,Write,NotebookEdit"])
        args.append("--dangerously-skip-permissions")
        args.append(prompt)
        return args

    def build_codex_command(self, prompt: str):
        if shutil.which("codex") is None:
            raise SystemExit("missing the required tool: codex")
        args = ["codex", "-c", "model_reasoning_effort=high", "--ask-for-approval", "never", "exec"]
        args.append("--skip-git-repo-check")
        args.extend(["--sandbox", "read-only"])
        args.extend(["--model", self.model])
        args.append(prompt)
        return args

    def run_agent(self, args: list[str]):
        """
        Run the judge with the patches directory as its working directory, so the diff is
        all it can reach. Its output is streamed for progress and captured for the result.
        """
        process = subprocess.Popen(args, cwd=self.patches.absolute().as_posix(), stdout=subprocess.PIPE, text=True)
        captured = []
        for line in process.stdout:
            sys.stdout.write(line)
            captured.append(line)
        if process.wait() != 0:
            raise SystemExit(f"{self.agent} exited {process.returncode} judging {self.task}/{self.rule} on {self.run}")
        return "".join(captured)

    def record(self, output: str):
        """
        Write the judge's answer into the snapshots tree, beside the challenge's own traces.
        The answer is stored exactly as the judge wrote it: the verdict and its justification
        are the judge's to format, and the path already says which rule and run they belong to.
        """
        self.result.parent.mkdir(parents=True, exist_ok=True)
        self.result.write_text(output.strip() + "\n")

    def evaluate(self):
        """
        Evaluate the rule, record the judge's answer, and return the verdict it opened with.
        """
        prompt = self.compose()
        match self.agent:
            case "claude":
                args = self.build_claude_command(prompt)
            case "codex":
                args = self.build_codex_command(prompt)
            case _:
                raise ValueError(f"unsupported agent: {self.agent}")
        output = self.run_agent(args)
        self.record(output)
        return verdict_of(output)


def discover_rules(validation: Path):
    """
    The rules of a validation, ordered R1, R2, ... as named by their files.
    """
    rules = []
    for rule in sorted(validation.glob("R*.md"), key=lambda path: int(path.stem[1:])):
        rules.append(rule.stem)
    return rules


def aggregate(results: dict, total: int):
    """
    The verdict over the isolated per-rule results. A verdict only carries meaning once
    every rule of the validation has been judged, so a run over a subset reports none.

    Nothing writes this to disk: the per-rule results are the record, and a stored verdict
    would go stale the moment a single rule is re-judged.
    """
    if len(results) < total:
        return f"n/a ({len(results)} of {total} rules judged)"
    if any(result is None for result in results.values()):
        return "UNRESOLVED"
    passed = sum(result == "PASS" for result in results.values())
    if passed == len(results):
        return "SATISFIED"
    if passed == 0:
        return "INCOMPLETE"
    return "PARTIAL"


def launch(patches: Path, agent: str, model: str, rules: list):
    """
    Judge every requested rule in its own session, recording each result as it lands.
    """
    validation, task, run = locate(patches)
    results = {}
    for rule in rules:
        judge = Judge(patches, agent, model, rule)
        print(f"==> judging {task.as_posix()} {rule} on {run}", file=sys.stderr)
        results[rule] = judge.evaluate()
        print(f"==> {results[rule] or 'UNRESOLVED'} ({judge.result.as_posix()})", file=sys.stderr)
    verdict = aggregate(results, len(discover_rules(validation)))
    if len(results) > 1:
        judged = ", ".join(f"{rule} {result or 'UNRESOLVED'}" for rule, result in results.items())
        print(f"==> {task.as_posix()} {run}: {judged} -> {verdict}", file=sys.stderr)
    return verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("patches", type=lambda s: s if Path(s).is_dir() else p.error(f"{s} is not a valid patches directory"))
    p.add_argument("agent", type=str, choices=["claude", "codex"])
    p.add_argument("model", type=str)
    p.add_argument("--rule", type=str, action="append", help="judge only this rule (repeatable); defaults to every rule")
    a = p.parse_args()
    patches = Path(a.patches)
    validation, task, run = locate(patches)
    rules = a.rule or discover_rules(validation)
    if not rules:
        p.error(f"{validation.as_posix()} has no rules")
    for rule in rules:
        if not Path(validation, f"{rule}.md").is_file():
            p.error(f"{validation.as_posix()} has no rule {rule}")
    launch(patches, a.agent, a.model, rules)
