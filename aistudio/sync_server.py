"""Upload the reviewed manifest through one existing ssctl SFTP session."""

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

import pexpect
import yaml


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/ossfs/workspace/QuaSiD"
MANIFEST = ROOT / "aistudio/SYNC_FILES.txt"
AUTH_HELPER = ROOT / "aistudio/connect_ai_studio.sh"
SSH_USER = "cairongyao.cry"
SSH_HOST = "127.0.0.1"
SSH_PORT = 58598
PROMPT = r"(?m)^sftp> "
RUNTIME_DIRS = {"experiments", "logs", "runs", "__pycache__", "datasets"}
RUNTIME_SUFFIXES = {".csv", ".npy", ".npz", ".pt", ".pth", ".log"}


def checked_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or value.startswith("-")
        or value == "."
    ):
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    return path


def prepare(deletions: list[str], files: list[str] | None = None) -> list[str]:
    manifest = MANIFEST.read_text(encoding="utf-8").splitlines()
    manifest_set = set(manifest)
    uploads = manifest if files is None else files
    upload_set = set(uploads)
    if (
        len(manifest) != len(manifest_set)
        or len(uploads) != len(upload_set)
        or len(deletions) != len(set(deletions))
    ):
        raise ValueError("Duplicate upload or deletion path")
    if any(value not in manifest_set for value in uploads):
        raise ValueError("Upload paths must be listed in aistudio/SYNC_FILES.txt")

    for value in uploads:
        path = checked_path(value)
        local = ROOT / value
        if not local.is_file() or local.resolve() != local.absolute():
            raise ValueError(f"Missing file or symlink in upload path: {value}")
        if path.parts[0] == "aistudio":
            raise ValueError(f"Local-only AI Studio file in upload manifest: {value}")
        if (
            set(path.parts) & RUNTIME_DIRS and not value.startswith("scripts/experiments/")
        ) or path.suffix.lower() in RUNTIME_SUFFIXES:
            raise ValueError(f"Runtime artifact in upload manifest: {value}")

    has_git = (ROOT / ".git").exists()
    if deletions and not has_git:
        raise ValueError("Cannot verify deletion history without a .git repository")

    for value in deletions:
        checked_path(value)
        if (ROOT / value).exists() or (ROOT / value).is_symlink() or value in upload_set:
            raise ValueError(f"Deletion path still exists locally: {value}")
        history = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", value], cwd=ROOT, text=True
        )
        if not history.strip():
            raise ValueError(f"Deletion path has no Git history: {value}")

    if has_git:
        subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
        subprocess.run(["git", "diff", "--cached", "--check"], cwd=ROOT, check=True)
    return uploads


def check_syntax(uploads: list[str]) -> list[str]:
    checked = []
    for value in uploads:
        path = ROOT / value
        suffix = path.suffix.lower()
        if suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=value)
        elif suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
        elif suffix in {".yaml", ".yml"}:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            continue
        checked.append(value)
    return checked


def read_password() -> str:
    # Read only the credential field in memory; never execute or log the local helper.
    source = AUTH_HELPER.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^\s*(?:export\s+)?(?:SSH_PASSWORD|PASSWORD|SSHPASS|AI_STUDIO_PASSWORD)=(.+)$",
        source,
    )
    if match is None:
        raise ValueError("Credential assignment not found in local authentication helper")
    values = shlex.split(match.group(1), comments=True)
    if len(values) != 1 or not values[0]:
        raise ValueError("Expected one literal password assignment")
    return values[0]


def missing_path(output: str) -> bool:
    # Accept only explicit missing-path diagnostics, never a general Failure or denial.
    lines = output.splitlines()
    return bool(lines) and all(
        "No such file or directory" in line or "not found" in line for line in lines
    )


class Sftp:
    def __init__(self, password: str) -> None:
        self.password = password
        self.child: pexpect.spawn | None = None
        self.target = "connect"
        self.command = "connect"
        self.response = ""

    def connect(self) -> None:
        options = [
            "PreferredAuthentications=password",
            "PubkeyAuthentication=no",
            "NumberOfPasswordPrompts=1",
            "BatchMode=no",
            "ProxyCommand=none",
            "ProxyJump=none",
            "ControlMaster=no",
            "ControlPath=none",
            "ClearAllForwardings=yes",
            "ConnectTimeout=15",
            "ConnectionAttempts=1",
            "StrictHostKeyChecking=yes",
        ]
        args = ["-q", "-F", "/dev/null", "-P", str(SSH_PORT)]
        for option in options:
            args.extend(["-o", option])
        args.append(f"{SSH_USER}@{SSH_HOST}")

        self.child = pexpect.spawn(
            "/usr/bin/sftp",
            args,
            cwd=str(ROOT),
            encoding="utf-8",
            timeout=30,
            dimensions=(24, 4096),
            env={**os.environ, "LC_ALL": "C", "TERM": "dumb", "SSH_ASKPASS_REQUIRE": "never"},
        )
        self.child.expect(r"(?i)password:")
        if not self.child.waitnoecho(timeout=5):
            raise RuntimeError("Password prompt did not disable echo")
        self.child.sendline(self.password)
        self.child.expect(PROMPT)
        self.child.setecho(False)
        print(f"Authenticated through {SSH_HOST}:{SSH_PORT}", flush=True)

    def run(self, command: str) -> str:
        assert self.child is not None
        self.command = command
        self.response = ""
        self.child.sendline(command)
        self.child.expect(PROMPT)
        lines = self.child.before.replace("\r", "").strip().splitlines()
        if lines and lines[0] == command:
            lines.pop(0)
        self.response = "\n".join(lines).strip()
        return self.response

    def silent(self, command: str) -> None:
        if self.run(command):
            raise RuntimeError("Unexpected SFTP response")

    def check_directory(self) -> None:
        output = self.run("pwd")
        prefix = "Remote working directory: "
        if not output.startswith(prefix):
            raise RuntimeError("Unrecognized SFTP pwd response")
        path = output[len(prefix):]
        if path != REMOTE and not path.startswith(REMOTE + "/"):
            raise RuntimeError("Remote directory resolves outside the project")

    def ensure_directory(self, path: str) -> None:
        self.target = path
        output = self.run(f'cd "{path}"')
        if output:
            if not missing_path(output):
                raise RuntimeError("Cannot enter remote directory")
            self.silent(f'mkdir "{path}"')
            self.silent(f'cd "{path}"')
        self.check_directory()
        self.silent(f'cd "{REMOTE}"')

    def ensure_hierarchy(self, leaf: str, verified: set[str]) -> None:
        # Probe the deepest relative directory first to skip redundant ancestor round-trips.
        self.target = leaf
        output = self.run(f'cd "{leaf}"')
        if not output:
            self.check_directory()
            self.silent(f'cd "{REMOTE}"')
            verified.add(leaf)
            verified.update(str(p) for p in PurePosixPath(leaf).parents if str(p) != ".")
            return
        if not missing_path(output):
            raise RuntimeError("Cannot enter remote directory")

        ancestors = [str(p) for p in reversed(PurePosixPath(leaf).parents) if str(p) != "."]
        for part in [*ancestors, leaf]:
            if part not in verified:
                self.ensure_directory(part)
                verified.add(part)

    def finish(self) -> None:
        assert self.child is not None
        self.target = "bye"
        self.command = "bye"
        self.child.sendline("bye")
        self.child.expect(pexpect.EOF)
        self.child.close()
        if self.child.exitstatus != 0:
            raise RuntimeError(f"SFTP exited with status {self.child.exitstatus}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--files", nargs="+", metavar="PATH", help="Select changed manifest files")
    scope.add_argument("--all", action="store_true", help="Select the complete manifest")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Check local syntax without imports or server access")
    mode.add_argument("--dry-run", action="store_true", help="Check the local plan only")
    parser.add_argument("--delete", action="append", default=[], metavar="OLD_PATH")
    args = parser.parse_args()

    started = time.monotonic()
    uploads = prepare(args.delete, args.files)
    if args.check_only:
        checked = check_syntax(uploads)
        summary = {"syntax_checked": checked, "seconds": round(time.monotonic() - started, 3)}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parents = sorted(
        {str(parent) for value in uploads for parent in PurePosixPath(value).parents if str(parent) != "."},
        key=lambda value: (len(PurePosixPath(value).parts), value),
    )
    plan = {
        "local_root": str(ROOT),
        "remote_root": REMOTE,
        "uploads": [
            {"local": str(ROOT / value), "remote": str(PurePosixPath(REMOTE) / value)}
            for value in uploads
        ],
        "deletions": args.delete,
        "directories": parents,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    leaf_parents = [
        path for path in parents
        if not any(other.startswith(path + "/") for other in parents)
    ]
    client = Sftp(read_password())
    report: dict[str, object] = {"uploaded": [], "deleted": [], "missing_deletions": []}
    uploaded: list[str] = []
    deleted: list[str] = []
    missing_deletions: list[str] = []
    try:
        client.connect()
        client.ensure_directory(REMOTE)
        verified_dirs: set[str] = set()
        for leaf in leaf_parents:
            client.ensure_hierarchy(leaf, verified_dirs)

        for old_path in args.delete:
            # Resolve existing parents before deletion to preserve the write boundary.
            client.target = old_path
            parent = str(PurePosixPath(old_path).parent)
            if parent != ".":
                output = client.run(f'cd "{parent}"')
                if output and missing_path(output):
                    missing_deletions.append(old_path)
                    continue
                if output:
                    raise RuntimeError("Cannot enter deletion parent")
                client.check_directory()
                client.silent(f'cd "{REMOTE}"')

            output = client.run(f'rm "{old_path}"')
            lines = [line for line in output.splitlines() if not line.startswith("Removing ")]
            if lines and missing_path("\n".join(lines)):
                missing_deletions.append(old_path)
            elif lines:
                raise RuntimeError("Cannot delete old remote file")
            else:
                deleted.append(old_path)
            print(f"Deletion handled: {old_path}", flush=True)

        for index, rel_path in enumerate(uploads, start=1):
            client.target = rel_path
            pending = f"{rel_path}.upload-{os.getpid()}"
            output = client.run(f'put "{rel_path}" "{pending}"')
            if any(not line.startswith("Uploading ") for line in output.splitlines()):
                raise RuntimeError("Unexpected upload response")
            # Same-directory rename replaces the inode; active readers keep their original file.
            client.silent(f'rename "{pending}" "{rel_path}"')
            uploaded.append(rel_path)
            print(f"Uploaded {index}/{len(uploads)}: {rel_path}", flush=True)

        client.finish()
        report["status"] = "complete"
    except (RuntimeError, OSError, pexpect.ExceptionPexpect) as error:
        report.update(status="failed", failed_path=client.target, command=client.command)
        if isinstance(error, pexpect.ExceptionPexpect):
            report["error"] = type(error).__name__
            response = (client.child.before if client.child is not None else "") or ""
        else:
            report["error"] = str(error)
            response = client.response
        report["response"] = response.replace(client.password, "[REDACTED]")
    finally:
        if client.child is not None and client.child.isalive():
            client.child.close(force=True)

    uploaded_set = set(uploaded)
    handled_set = set(deleted) | set(missing_deletions)
    report["uploaded"] = uploaded
    report["deleted"] = deleted
    report["missing_deletions"] = missing_deletions
    report["not_uploaded"] = [value for value in uploads if value not in uploaded_set]
    report["not_deleted"] = [value for value in args.delete if value not in handled_set]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
