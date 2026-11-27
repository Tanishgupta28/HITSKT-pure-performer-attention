"""Verify and install the private Junyi runtime files without printing their contents."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import JUNYI_ASSETS
from app.junyi_bank import BANK_SHA256


ARCHIVE_SHA256 = "e50e501cdc19b20babc85963656604a99d16b024b86557a8683bd7ab57fdf62b"
BANK = "junyi_hitskt_question_bank.csv"
HISTORY = "student1_pre_test_history.json"
MANIFEST = "bundle_manifest.json"


def _declared_hash(manifest, name):
    if isinstance(manifest, list):
        return next((found for item in manifest if (found := _declared_hash(item, name))), None)
    if not isinstance(manifest, dict):
        return None
    for key, value in manifest.items():
        if PurePosixPath(key).name == name:
            return value.get("sha256") if isinstance(value, dict) else value
    filename = manifest.get("name", manifest.get("path", manifest.get("file")))
    if isinstance(filename, str) and PurePosixPath(filename).name == name:
        return manifest.get("sha256")
    return next((found for value in manifest.values() if (found := _declared_hash(value, name))), None)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("--passphrase-file", required=True, type=Path)
    cli.add_argument("--gpg", default="gpg")
    args = cli.parse_args()
    if args.archive.stat().st_size != 182_158 or hashlib.sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        cli.error("The encrypted archive does not match the verified Junyi v1 release.")
    decrypted = subprocess.run([args.gpg, "--batch", "--pinentry-mode", "loopback", "--no-symkey-cache", "--passphrase-file",
                                str(args.passphrase_file.resolve()), "--decrypt", str(args.archive.resolve())],
                               capture_output=True, timeout=30)
    if decrypted.returncode:
        cli.error("The archive could not be decrypted using the supplied private passphrase file.")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(decrypted.stdout), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.name not in {BANK, HISTORY, MANIFEST}:
                continue
            if not member.isfile() or member.size > 3_000_000 or path.name in files:
                cli.error("The archive contains an invalid runtime member.")
            with archive.extractfile(member) as file:
                files[path.name] = file.read()
    if files.keys() != {BANK, HISTORY, MANIFEST}:
        cli.error("A required runtime member is missing.")
    manifest = json.loads(files[MANIFEST])
    history_hash = hashlib.sha256(files[HISTORY]).hexdigest()
    if hashlib.sha256(files[BANK]).hexdigest() != BANK_SHA256:
        cli.error("Unexpected question bank hash in the verified archive.")
    declared_history_hash = _declared_hash(manifest, HISTORY)
    if declared_history_hash is not None and declared_history_hash != history_hash:
        cli.error("The history member does not match the bundle manifest.")
    JUNYI_ASSETS.mkdir(parents=True, exist_ok=True)
    for name in (BANK, HISTORY, MANIFEST):
        temporary = JUNYI_ASSETS / f"{name}.tmp"
        temporary.write_bytes(files[name])
        temporary.replace(JUNYI_ASSETS / name)
    print(f"Verified private bundle installed. Set JUNYI_HISTORY_SHA256={history_hash} in the local backend environment.")


if __name__ == "__main__":
    main()
