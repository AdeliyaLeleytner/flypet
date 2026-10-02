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

    def test_preview_export_keeps_a_consistent_brain_only_manifest(self):
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            (bundle / "fixture.json").write_text("{}")
            (bundle / "manifest.json").write_text(
                json.dumps({"files": {"fixture.json": file_hash(bundle / "fixture.json")}})
            )
            out = Path(tmp) / "preview"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "release/build_latent_space.py"),
                    "--bundle",
                    str(bundle),
                    "--out",
                    str(out),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            manifest = json.loads((out / "data/manifest.json").read_text())
            fetch_data.audit_manifest(manifest)
            self.assertEqual({f["group"] for f in manifest["files"]}, {"brain"})
            self.assertNotIn("archives", manifest)
            self.assertNotIn("research_coverage", manifest)

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


class SharedSourceFileTests(unittest.TestCase):
    def test_package_materializes_shared_source_inodes(self):
        import os
        from release import package

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data/a").write_bytes(b"{}")
            os.link(root / "data/a", root / "data/b")
            entries = [
                {"path": name, "group": "g", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
                for name in ("data/a", "data/b")
            ]
            (root / "data/manifest.json").write_text(json.dumps({"files": entries}))
            archive = root / "test.tar.gz"
            with (
                patch.object(package, "ROOT", root),
                patch.object(fetch_data, "ROOT", root),
                patch("sys.argv", ["package", "g", "--out", str(archive)]),
            ):
                package.main()
            with tarfile.open(archive, "r:gz") as stream:
                self.assertTrue(all(m.isfile() for m in stream))
            destination = root / "consumer"
            destination.mkdir()
            with patch.object(fetch_data, "ROOT", destination):
                fetch_data.extract_archive(archive, entries)
            self.assertNotEqual(
                (destination / "data/a").stat().st_ino, (destination / "data/b").stat().st_ino
            )
            self.assertEqual((destination / "data/b").read_bytes(), b"{}")

    def test_even_internal_archive_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "links.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                regular = tarfile.TarInfo("data/a")
                regular.size = 2
                stream.addfile(regular, io.BytesIO(b"{}"))
                alias = tarfile.TarInfo("data/b")
                alias.type = tarfile.LNKTYPE
                alias.linkname = "data/a"
                stream.addfile(alias)
            entries = [
                {"path": name, "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
                for name in ("data/a", "data/b")
            ]
            with patch.object(fetch_data, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Unexpected archive member"):
                    fetch_data.extract_archive(archive, entries)
            self.assertFalse((root / "data/a").exists())


class ResearchReleaseTests(unittest.TestCase):
    def test_two_archives_one_group_download_and_install_without_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "remote"
            source.mkdir()
            files, archives = [], []
            for i in range(2):
                name = f"part-{i}.tar.gz"
                member = f"data/result-{i}.json"
                content = str(i).encode()
                archive = source / name
                with tarfile.open(archive, "w:gz") as stream:
                    info = tarfile.TarInfo(member)
                    info.size = len(content)
                    stream.addfile(info, io.BytesIO(content))
                files.append(
                    {
                        "path": member,
                        "group": "family",
                        "archive": name,
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
                archives.append(
                    {
                        "name": name,
                        "group": "family",
                        "url": archive.as_uri(),
                        "size": archive.stat().st_size,
                        "sha256": file_hash(archive),
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "files": files,
                        "archives": archives,
                        "group_aliases": {"research": ["family"]},
                    }
                )
            )
            with (
                patch.object(fetch_data, "ROOT", root),
                patch.object(fetch_data, "MANIFEST", manifest),
                patch("sys.argv", ["fetch", "research", "--no-cache"]),
            ):
                fetch_data.main()
            self.assertEqual((root / "data/result-0.json").read_text(), "0")
            self.assertEqual((root / "data/result-1.json").read_text(), "1")
            self.assertEqual(list((root / "data/.downloads").iterdir()), [])
            files[0].pop("archive")
            with self.assertRaisesRegex(ValueError, "Multipart"):
                fetch_data.archive_entries({"files": files, "archives": archives}, archives[0])

    def test_metadata_pickle_rewrite_preserves_values_for_old_and_framed_protocols(self):
        import pickle
        from release.research import Sanitizer

        private = "/" + "Users" + "/fixture/project/checkpoint.pt"
        sanitizer = Sanitizer(Path("/unused-project"))
        for protocol in (2, 4, 5):
            value = {"path": private, "weights": [1.5, -2.0, 0.0], "seed": 713}
            original = pickle.dumps(value, protocol=protocol)
            cleaned = sanitizer.pickle(original)
            restored = pickle.loads(cleaned)  # This fixture was constructed locally above.
            self.assertNotEqual(restored["path"], private)
            self.assertEqual(restored["weights"], value["weights"])
            self.assertEqual(restored["seed"], 713)

    def test_tensor_storage_and_numeric_npz_members_are_byte_identical(self):
        import pickle
        import zipfile
        import numpy as np
        from release.research import Sanitizer

        private = "/" + "Users" + "/fixture/project/run"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sanitizer = Sanitizer(root / "source")
            source, target = root / "source.pt", root / "target.pt"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("checkpoint/data.pkl", pickle.dumps({"path": private}, protocol=2))
                archive.writestr("checkpoint/data/0", b"unchanged tensor storage")
            self.assertTrue(sanitizer.file(source, target))
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.read("checkpoint/data/0"), b"unchanged tensor storage")
                self.assertNotIn(private, pickle.loads(archive.read("checkpoint/data.pkl"))["path"])
            source, target = root / "source.npz", root / "target.npz"
            np.savez_compressed(
                source, weights=np.arange(20, dtype=np.float32), path=np.array([private])
            )
            self.assertTrue(sanitizer.file(source, target))
            with zipfile.ZipFile(source) as old, zipfile.ZipFile(target) as new:
                self.assertEqual(old.read("weights.npy"), new.read("weights.npy"))
            np.testing.assert_equal(np.load(target)["weights"], np.arange(20, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
