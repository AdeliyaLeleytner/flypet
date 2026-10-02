"""Review export scope, integrity, privacy and external-data staging checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import pickle
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "deploy" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle = load_helper("build_paper_bundle")
stage = load_helper("stage_paper_data")


class BundleTests(unittest.TestCase):
    def test_export_normalizes_metadata_paths_without_changing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = {
                "physics_run": str(root / "paper/results/cases-v1"),
                "reader": {"checkpoint_path": str(root / "data/model.pt"), "sha256": "a" * 64},
                "claude_cli": str(root / "bin/claude"),
                "prompt": "Hello fly.",
            }
            exported, changed = bundle.export_json_paths(data, root)
            self.assertEqual(exported["physics_run"], "paper/results/cases-v1")
            self.assertEqual(
                exported["reader"]["checkpoint_path"], "external-models/brain_projector.pt"
            )
            self.assertEqual(exported["claude_cli"], "claude")
            self.assertEqual(exported["reader"]["sha256"], data["reader"]["sha256"])
            self.assertEqual(exported["prompt"], data["prompt"])
            self.assertTrue(data["physics_run"].startswith(str(root)))
            self.assertEqual(len(changed), 3)
            included, _ = bundle.export_json_paths(data, root, include_reader_checkpoint=True)
            self.assertEqual(included["reader"]["checkpoint_path"], bundle.READER_TARGET)

    def test_checkpoint_export_accepts_only_pinned_active_seed_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / bundle.READER_SOURCE
            source.parent.mkdir(parents=True)
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("checkpoint/data.pkl", pickle.dumps({"model": "Qwen/Qwen3-0.6B"}))
            (source.parent / "SHA256.json").write_text(
                json.dumps({"brain_projector.pt": bundle.digest(source)})
            )
            active = root / bundle.READER_TARGET
            active.parent.mkdir(parents=True)
            active.symlink_to(source)
            selected, metadata = bundle.reader_checkpoint(root)
            self.assertEqual(selected, source)
            self.assertEqual(metadata["seed"], 0)
            self.assertFalse(metadata["other_seed_checkpoints_included"])
            active.unlink()
            other = root / "other.pt"
            other.write_bytes(source.read_bytes())
            active.symlink_to(other)
            with self.assertRaises(ValueError):
                bundle.reader_checkpoint(root)

    def test_sensitive_path_rejected_without_rewriting_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "note.md"
            content = "/" + "home" + "/" + "example-user" + "/private-record"
            p.write_text(content)
            with self.assertRaises(ValueError):
                bundle.inspect_file(p, Path("note.md"))
            self.assertEqual(p.read_text(), content)

    def test_checksum_validation_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "safe.txt").write_text("expected")
            manifest = {"files": {"safe.txt": {"sha256": bundle.digest(p / "safe.txt")}}}
            (p / "bundle-manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(bundle.verify_tree(p)["ok"])
            (p / "safe.txt").write_text("changed")
            self.assertFalse(bundle.verify_tree(p)["ok"])

    def test_stage_copies_only_declared_inputs_and_preserves_private_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            (source / "data").mkdir(parents=True)
            target.mkdir()
            (source / "data/input.dat").write_bytes(b"dataset")
            (source / "data/memory.npz").write_bytes(b"private")
            manifest = {
                "files": {"data/input.dat": {"sha256": stage.digest(source / "data/input.dat")}}
            }
            (target / "external-data.json").write_text(json.dumps(manifest))
            result = stage.stage(source, target)
            self.assertEqual(result["n_assets"], 1)
            self.assertEqual((target / "data/input.dat").read_bytes(), b"dataset")
            self.assertFalse((target / "data/memory.npz").exists())
            self.assertEqual((source / "data/memory.npz").read_bytes(), b"private")

    def test_stage_rejects_wrong_hash_and_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            source, target = p / "source", p / "target"
            (source / "data").mkdir(parents=True)
            target.mkdir()
            (source / "data/input.dat").write_bytes(b"dataset")
            for relative in ("data/input.dat", "../escape"):
                (target / "external-data.json").write_text(
                    json.dumps({"files": {relative: {"sha256": "wrong"}}})
                )
                with self.assertRaises(ValueError):
                    stage.stage(source, target)
            self.assertFalse((target / "data/input.dat").exists())


if __name__ == "__main__":
    unittest.main()
