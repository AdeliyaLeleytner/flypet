"""Artifact integrity checks use only tiny local fixtures."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from flypet.neural_records import file_hash, verify_bundle
from scripts import fetch_data


class BundleTests(unittest.TestCase):
    def test_complete_bundle_and_tampered_or_unlisted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "weights.safetensors"
            artifact.write_bytes(b"fixture")
            manifest = {"files": {artifact.name: file_hash(artifact)}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_bundle(root), manifest)
            (root / "private.log").write_text("unlisted")
            with self.assertRaisesRegex(ValueError, "unlisted"):
                verify_bundle(root)
            (root / "private.log").unlink()
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_bundle(root)

    def test_traversal_and_external_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "bundle"
            root.mkdir()
            outside = parent / "outside"
            outside.write_bytes(b"fixture")
            for name in ("../outside", str(outside), "link"):
                if name == "link":
                    (root / name).symlink_to(outside)
                (root / "manifest.json").write_text(
                    json.dumps({"files": {name: file_hash(outside)}})
                )
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    verify_bundle(root)


class ArchiveTests(unittest.TestCase):
    def entry(self, name="data/result.json", content=b"{}"):
        return {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def archive(self, root, members):
        path = root / "archive.tar.gz"
        with tarfile.open(path, "w:gz") as stream:
            for name, content in members:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                stream.addfile(member, io.BytesIO(content))
        return path

    def test_checked_install_and_local_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fetch_data, "ROOT", Path(tmp)):
            root = Path(tmp)
            archive = self.archive(root, [("data/result.json", b"{}")])
            fetch_data.extract_archive(archive, [self.entry()])
            destination = root / "data/result.json"
            self.assertEqual(destination.read_bytes(), b"{}")
            destination.write_bytes(b"local research")
            with self.assertRaisesRegex(ValueError, "Local file differs"):
                fetch_data.extract_archive(archive, [self.entry()])
            self.assertEqual(destination.read_bytes(), b"local research")
            fetch_data.extract_archive(archive, [self.entry()], force=True)
            self.assertEqual(destination.read_bytes(), b"{}")

    def test_unexpected_duplicate_missing_or_corrupt_members_leave_no_output(self):
        variants = [
            [("../escape", b"{}")],
            [("data/result.json", b"{}"), ("data/result.json", b"{}")],
            [],
            [("data/result.json", b"[]")],
        ]
        for members in variants:
            with self.subTest(members=members), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(fetch_data, "ROOT", root):
                    archive = self.archive(root, members)
                    with self.assertRaises(ValueError):
                        fetch_data.extract_archive(archive, [self.entry()])
                    self.assertFalse((root / "data/result.json").exists())

    def test_destination_symlink_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / "repository"
            root.mkdir()
            (root / "data").symlink_to(parent, target_is_directory=True)
            with patch.object(fetch_data, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    fetch_data.safe_destination("data/result.json")


if __name__ == "__main__":
    unittest.main()
