import argparse
import json
import os
import secrets
import shlex
import shutil
import subprocess
from pathlib import Path

WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
UV_CACHE_DIR = Path("workspace/uv-cache").resolve()
UV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME = Path("workspace/hf-home").resolve()
HF_HOME.mkdir(parents=True, exist_ok=True)
HF_TOKEN_PATH = Path(Path.home(), ".cache/huggingface/token")

os.environ.setdefault("UV_CACHE_DIR", UV_CACHE_DIR.as_posix())
os.environ.setdefault("HF_HOME", HF_HOME.as_posix())
os.environ.setdefault("HF_TOKEN_PATH", HF_TOKEN_PATH.as_posix())
os.environ.setdefault("PYTHONUNBUFFERED", "1")

for tool in ("uv",):
    if shutil.which(tool) is None:
        raise SystemExit(f"missing the required tool: {tool}")


class Runner:

    def __init__(self, framework: str, challenge: str, agent: str, model: str):
        self.framework, self.challenge, self.agent, self.model = framework, challenge, agent, model
        self.uuid = "-".join([framework, secrets.token_hex(3)])
        self.workspace = Path(WORKSPACE, challenge, self.uuid)
        self.workspace.mkdir(parents=True, exist_ok=False)
        self.snapshot = Path("snapshots", challenge, self.uuid)

    def prepare(self):
        # Materialize the workspace: the per-challenge script clones, patches, and builds the framework.
        Path(self.workspace, "artifacts").mkdir()
        prepare = Path(self.challenge, "prepare", f"{self.framework}.sh").as_posix()
        subprocess.run(["bash", prepare, self.workspace.as_posix()], check=True)

    def attempt(self):
        # Run the agent in the workspace; stream its JSON events to stdout for progress.
        instruction = Path(self.challenge, "instruction.md").read_text()
        instruction = instruction.format(framework=self.framework)
        match self.agent:
            case "claude":
                args = self.build_claude_command(instruction)
            case "codex":
                args = self.build_codex_command(instruction)
            case _:
                raise ValueError(f"unsupported agent: {self.agent}")
        self.run_agent(args)

    def build_claude_command(self, instruction: str):
        if shutil.which("claude") is None:
            raise SystemExit("missing the required tool: claude")
        args = ["claude", "--print"]
        args.extend(["--model", self.model, "--effort", "xhigh"])
        args.extend(["--output-format", "stream-json", "--include-partial-messages"])
        # question-and-answer challenges are read-only: the agent investigates the code, never edits it.
        # Keep --disallowedTools ahead of other flags so its variadic value never swallows the instruction.
        if "question-and-answer" in self.challenge:
            args.extend(["--disallowedTools", "Edit,Write,NotebookEdit"])
        args.extend(["--dangerously-skip-permissions", "--verbose"])
        args.append(instruction)
        return args

    def build_codex_command(self, instruction: str):
        if shutil.which("codex") is None:
            raise SystemExit("missing the required tool: codex")
        sandbox = "read-only" if "question-and-answer" in self.challenge else "workspace-write"
        args = ["codex", "-c", "model_reasoning_effort=high", "--ask-for-approval", "never", "exec", "--json"]
        args.append("--skip-git-repo-check")
        args.extend(["--sandbox", sandbox])
        args.extend(["--model", self.model])
        args.append(instruction)
        return args

    def run_agent(self, args):
        script = Path(self.workspace, f"{self.agent}-launch.sh")
        with script.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {shlex.quote(self.workspace.as_posix())}\n")
            f.write(f"exec {' '.join(shlex.quote(str(arg)) for arg in args)}\n")
        script.chmod(0o755)
        subprocess.run(["bash", script.as_posix()], check=True)

    def capture(self):
        snapshot = self.snapshot
        snapshot.mkdir(parents=True, exist_ok=True)
        patches = Path(snapshot, "patches")
        patches.mkdir(parents=True, exist_ok=True)
        # Capture a patch for each modified codebase; skip the ones the agent left untouched.
        for codebase in sorted(git.parent for git in self.workspace.glob("*/.git")):
            subprocess.run(["git", "add", "-A"], cwd=codebase, check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet", "main"], cwd=codebase, check=False).returncode == 0:
                continue
            with Path(patches, f"{codebase.name}.patch").open("wb") as patch:
                subprocess.run(["git", "diff", "--cached", "--binary", "main"], cwd=codebase, stdout=patch, check=True)
        # Capture the launch script, artifacts, and agent session.
        script = Path(self.workspace, f"{self.agent}-launch.sh")
        if script.exists():
            shutil.copy2(script, Path(snapshot, script.name))
        shutil.copytree(Path(self.workspace, "artifacts"), Path(snapshot, "artifacts"), dirs_exist_ok=True)
        match self.agent:
            case "claude":
                self.capture_claude_sessions()
            case "codex":
                self.capture_codex_sessions()
            case _:
                raise ValueError(f"unsupported agent: {self.agent}")

    def capture_claude_sessions(self):
        project = self.claude_project()
        if project.exists():
            shutil.copytree(project, Path(self.snapshot, "sessions"), dirs_exist_ok=True)

    def capture_codex_sessions(self):
        destination = Path(self.snapshot, "sessions")
        for transcript in self.codex_sessions():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(transcript, Path(destination, transcript.name))

    def claude_project(self):
        workspace = self.workspace.as_posix().replace("/", "-")
        return Path(Path.home(), ".claude/projects", workspace)

    def codex_sessions(self):
        workspace = self.workspace.resolve()
        for transcript in sorted(Path(Path.home(), ".codex", "sessions").glob("**/*.jsonl")):
            try:
                with transcript.open(encoding="utf-8") as f:
                    metadata = json.loads(f.readline())
            except (OSError, ValueError):
                continue
            cwd = (metadata.get("payload") or {}).get("cwd")
            if cwd and Path(cwd).resolve() == workspace:
                yield transcript

    def cleanup(self):
        # Remove the workspace and the agent's native session store.
        shutil.rmtree(self.workspace, ignore_errors=True)
        match self.agent:
            case "claude":
                shutil.rmtree(self.claude_project(), ignore_errors=True)
            case "codex":
                for transcript in self.codex_sessions():
                    transcript.unlink(missing_ok=True)

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
    p.add_argument("challenge", type=lambda s: s if Path(s).is_dir() else p.error(f"{s} is not a valid task"))
    p.add_argument("agent", type=str, choices=["claude", "codex"])
    p.add_argument("model", type=str)
    a = p.parse_args()
    Runner(a.framework, a.challenge, a.agent, a.model).launch()
