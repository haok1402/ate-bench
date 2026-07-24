import argparse
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
UV_CACHE_DIR = Path("workspace/uv-cache").resolve()
UV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME = Path("workspace/hf-home").resolve()
HF_HOME.mkdir(parents=True, exist_ok=True)
HF_TOKEN_PATH = Path(Path.home(), ".cache/huggingface/token")
DEFAULT_MODELS = {"claude": "claude-opus-4-7", "codex": "gpt-5.6-sol"}

os.environ.setdefault("UV_CACHE_DIR", UV_CACHE_DIR.as_posix())
os.environ.setdefault("HF_HOME", HF_HOME.as_posix())
os.environ.setdefault("HF_TOKEN_PATH", HF_TOKEN_PATH.as_posix())
os.environ.setdefault("PYTHONUNBUFFERED", "1")

for tool in ("uv",):
    if shutil.which(tool) is None:
        raise SystemExit("required tool not on PATH: %s" % tool)


class Runner:

    def __init__(self, framework: str, challenge: str, agent: str, model: str | None):
        self.framework, self.challenge, self.agent, self.model = framework, challenge, agent, model
        self.uuid = "-".join([framework, secrets.token_hex(3)])
        self.workspace = Path(WORKSPACE, challenge, self.uuid)
        self.workspace.mkdir(parents=True, exist_ok=False)

    def prepare(self):
        # Materialize the workspace: the per-challenge script clones, patches, and builds the framework.
        Path(self.workspace, "artifacts").mkdir()
        prepare = Path(self.challenge, "prepare", "%s.sh" % self.framework).as_posix()
        subprocess.run(["bash", prepare, self.workspace.as_posix()], check=True)

    def attempt(self):
        # Run the agent in the workspace; stream its JSON events to stdout for progress.
        instruction = Path(self.challenge, "instruction.md").read_text()
        instruction = instruction.format(framework=self.framework)
        if self.agent == "claude":
            args = self.claude_args(instruction)
        elif self.agent == "codex":
            args = self.codex_args(instruction)
        else:
            raise ValueError("unsupported agent: %s" % self.agent)
        self.run_agent(args)

    def claude_args(self, instruction: str):
        if shutil.which("claude") is None:
            raise SystemExit("required tool not on PATH: claude")
        args = ["claude", "--print"]
        args.extend(["--model", self.model or DEFAULT_MODELS["claude"], "--effort", "xhigh"])
        args.extend(["--output-format", "stream-json", "--include-partial-messages"])
        # question-and-answer challenges are read-only: the agent investigates the code, never edits it.
        # Keep --disallowedTools ahead of other flags so its variadic value never swallows the instruction.
        if "question-and-answer" in self.challenge:
            args.extend(["--disallowedTools", "Edit,Write,NotebookEdit"])
        args.extend(["--dangerously-skip-permissions", "--verbose"])
        args.append(instruction)
        return args

    def codex_args(self, instruction: str):
        if shutil.which("codex") is None:
            raise SystemExit("required tool not on PATH: codex")
        sandbox = "read-only" if "question-and-answer" in self.challenge else "workspace-write"
        args = ["codex", "-c", "model_reasoning_effort=high", "--ask-for-approval", "never", "exec", "--json"]
        args.append("--skip-git-repo-check")
        args.extend(["--sandbox", sandbox])
        args.extend(["--model", self.model or DEFAULT_MODELS["codex"]])
        args.append(instruction)
        return args

    def run_agent(self, args):
        script = Path(self.workspace, "artifacts", "%s-launch.sh" % self.agent)
        with script.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write("cd %s\n" % shlex.quote(self.workspace.as_posix()))
            f.write("exec %s\n" % " ".join(shlex.quote(str(arg)) for arg in args))
        script.chmod(0o755)
        subprocess.run(["bash", script.as_posix()], check=True)

    def capture(self):
        snapshot = Path("snapshots", self.challenge, self.uuid)
        snapshot.mkdir(parents=True, exist_ok=True)
        patches = Path(snapshot, "patches")
        patches.mkdir(parents=True, exist_ok=True)
        # Capture a patch for each modified codebase; skip the ones the agent left untouched.
        for codebase in sorted(git.parent for git in self.workspace.glob("*/.git")):
            subprocess.run(["git", "add", "-A"], cwd=codebase, check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet", "main"], cwd=codebase).returncode == 0:
                continue
            with Path(patches, "%s.patch" % codebase.name).open("wb") as patch:
                subprocess.run(["git", "diff", "--cached", "--binary", "main"], cwd=codebase, stdout=patch, check=True)
        # Capture the artifacts and any agent-specific session directory.
        shutil.copytree(Path(self.workspace, "artifacts"), Path(snapshot, "artifacts"), dirs_exist_ok=True)
        if self.agent == "claude":
            project = Path(Path.home(), ".claude/projects", self.workspace.as_posix().replace("/", "-"))
            if project.exists():
                shutil.copytree(project, Path(snapshot, "claude-session"), dirs_exist_ok=True)
        elif self.agent == "codex":
            self.capture_codex_sessions(Path(snapshot, "codex-sessions"))

    def capture_codex_sessions(self, destination: Path):
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
        sessions = Path(codex_home, "sessions")
        if not sessions.exists():
            print("warning: Codex session directory not found: %s" % sessions, file=sys.stderr)
            return

        copied = 0
        workspace = self.workspace.resolve()
        for transcript in sorted(sessions.rglob("*.jsonl")):
            try:
                with transcript.open(encoding="utf-8") as session:
                    metadata = json.loads(session.readline())
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue

            payload = metadata.get("payload")
            if metadata.get("type") != "session_meta" or not isinstance(payload, dict):
                continue
            session_workspace = payload.get("cwd")
            if not isinstance(session_workspace, str):
                continue
            if Path(session_workspace).resolve() != workspace:
                continue

            target = Path(destination, transcript.relative_to(sessions))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(transcript, target)
            copied += 1

        if not copied:
            print("warning: no Codex sessions found for workspace: %s" % workspace, file=sys.stderr)

    def cleanup(self):
        # Remove the workspace and any Claude Code session.
        shutil.rmtree(self.workspace, ignore_errors=True)
        if self.agent == "claude":
            project = Path(Path.home(), ".claude/projects", self.workspace.as_posix().replace("/", "-"))
            shutil.rmtree(project, ignore_errors=True)

    def launch(self):
        self.prepare()
        try:
            self.attempt()
        finally:
            # If the attempt failed, we still capture. If the capture failed, we never cleanup.
            # This allows manual inspection over failures at different points of the execution.
            self.capture()
            self.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("framework", type=str, choices=["torchtitan", "pith-train", "Megatron-LM"])
    p.add_argument("challenge", type=lambda s: s if Path(s).is_dir() else p.error("%s is not a valid task" % s))
    p.add_argument("--agent", type=str, choices=["claude", "codex"], default="claude")
    p.add_argument("--model", type=str, default=None, help="Agent model override")
    a = p.parse_args()
    Runner(a.framework, a.challenge, a.agent, a.model).launch()
