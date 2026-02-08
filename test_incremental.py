"""
Integration tests for incremental upload logic in upload.py.
Uses real ChromaDB (localhost:9000) and temporary git repos.
Run: pytest test_incremental.py -v
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest

import upload


CHROMA_HOST = "localhost"
CHROMA_PORT = 9000
TEST_COLLECTION = f"test-incremental-{int(time.time())}"


@pytest.fixture
def client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@pytest.fixture
def collection(client):
    """Create a throwaway collection, clean up after."""
    name = TEST_COLLECTION
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.get_or_create_collection(name=name)
    yield coll
    try:
        client.delete_collection(name)
    except Exception:
        pass


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with a few Python files."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)

    # Initial commit with two files
    (repo / "main.py").write_text("def hello():\n    print('hello')\n")
    (repo / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True)

    return repo


@pytest.fixture
def manifest(tmp_path):
    """Provide a temporary manifest path and patch upload to use it."""
    manifest_path = tmp_path / ".chroma-uploads.json"
    with patch.object(upload, "append_to_manifest") as mock_append:
        # Make append_to_manifest write to our temp manifest
        def _append(input_path, collection_name, processed, total_chunks, git_meta, perf=None):
            entry = {
                "collection": collection_name,
                "repo": git_meta.get("git_project_name"),
                "git_commit": git_meta.get("git_commit_hash"),
                "git_branch": git_meta.get("git_branch"),
                "files": processed,
                "chunks": total_chunks,
                "input_path": input_path,
            }
            with open(manifest_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        mock_append.side_effect = _append
        yield manifest_path


def get_commit(repo_path):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
    ).strip()


def write_manifest(manifest_path, entry):
    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --- get_changed_files ---


class TestGetChangedFiles:
    def test_added_file(self, git_repo):
        old_commit = get_commit(git_repo)
        (git_repo / "new_file.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add file"], cwd=git_repo, capture_output=True)
        new_commit = get_commit(git_repo)

        result = upload.get_changed_files(str(git_repo), old_commit, new_commit, "source-code")

        assert len(result["added"]) == 1
        assert result["added"][0].endswith("new_file.py")
        assert result["modified"] == []
        assert result["deleted"] == []

    def test_modified_file(self, git_repo):
        old_commit = get_commit(git_repo)
        (git_repo / "main.py").write_text("def hello():\n    print('changed')\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "modify"], cwd=git_repo, capture_output=True)
        new_commit = get_commit(git_repo)

        result = upload.get_changed_files(str(git_repo), old_commit, new_commit, "source-code")

        assert result["added"] == []
        assert len(result["modified"]) == 1
        assert result["modified"][0].endswith("main.py")
        assert result["deleted"] == []

    def test_deleted_file(self, git_repo):
        old_commit = get_commit(git_repo)
        (git_repo / "utils.py").unlink()
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "delete"], cwd=git_repo, capture_output=True)
        new_commit = get_commit(git_repo)

        result = upload.get_changed_files(str(git_repo), old_commit, new_commit, "source-code")

        assert result["added"] == []
        assert result["modified"] == []
        assert len(result["deleted"]) == 1
        assert result["deleted"][0].endswith("utils.py")

    def test_renamed_file(self, git_repo):
        old_commit = get_commit(git_repo)
        (git_repo / "utils.py").rename(git_repo / "helpers.py")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "rename"], cwd=git_repo, capture_output=True)
        new_commit = get_commit(git_repo)

        result = upload.get_changed_files(str(git_repo), old_commit, new_commit, "source-code")

        assert any("helpers.py" in f for f in result["added"])
        assert any("utils.py" in f for f in result["deleted"])

    def test_filters_by_store_type(self, git_repo):
        old_commit = get_commit(git_repo)
        (git_repo / "readme.md").write_text("# Hello\n")
        (git_repo / "new_code.py").write_text("y = 2\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "mixed"], cwd=git_repo, capture_output=True)
        new_commit = get_commit(git_repo)

        result = upload.get_changed_files(str(git_repo), old_commit, new_commit, "source-code")

        added_names = [os.path.basename(f) for f in result["added"]]
        assert "new_code.py" in added_names
        assert "readme.md" not in added_names

    def test_cross_branch_diff(self, git_repo):
        """git diff works across branches — key to incremental branch handling."""
        old_commit = get_commit(git_repo)
        subprocess.run(["git", "checkout", "-b", "feature"], cwd=git_repo, capture_output=True)
        (git_repo / "feature.py").write_text("def feat(): pass\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "feature"], cwd=git_repo, capture_output=True)
        new_commit = get_commit(git_repo)

        result = upload.get_changed_files(str(git_repo), old_commit, new_commit, "source-code")

        assert len(result["added"]) == 1
        assert result["added"][0].endswith("feature.py")


# --- delete_file_chunks ---


class TestDeleteFileChunks:
    def test_deletes_matching_chunks(self, collection):
        # Seed collection with chunks from two files
        collection.add(
            ids=["a_001", "a_002", "b_001"],
            documents=["chunk a1", "chunk a2", "chunk b1"],
            metadatas=[
                {"file_path": "/repo/a.py"},
                {"file_path": "/repo/a.py"},
                {"file_path": "/repo/b.py"},
            ],
        )
        assert collection.count() == 3

        deleted = upload.delete_file_chunks(collection, ["/repo/a.py"])

        assert deleted == 1
        assert collection.count() == 1
        remaining = collection.get(include=["metadatas"])
        assert remaining["metadatas"][0]["file_path"] == "/repo/b.py"

    def test_no_paths_is_noop(self, collection):
        collection.add(ids=["x"], documents=["doc"], metadatas=[{"file_path": "/x.py"}])
        deleted = upload.delete_file_chunks(collection, [])
        assert deleted == 0
        assert collection.count() == 1

    def test_nonexistent_paths_no_error(self, collection):
        collection.add(ids=["x"], documents=["doc"], metadatas=[{"file_path": "/x.py"}])
        deleted = upload.delete_file_chunks(collection, ["/no/such/file.py"])
        assert deleted == 1  # paths processed, even if no matches
        assert collection.count() == 1  # original untouched


# --- get_upload_plan ---


class TestGetUploadPlan:
    def test_no_git_meta_returns_full(self, client):
        files = ["/some/file.py"]
        plan = upload.get_upload_plan(client, "whatever", {}, files, "source-code", "/some")
        assert plan["action"] == "full"
        assert plan["files"] == files

    def test_no_manifest_returns_full(self, client, git_repo, tmp_path):
        git_meta = upload.get_git_metadata(str(git_repo))
        files = [str(git_repo / "main.py")]
        manifest_path = tmp_path / ".chroma-uploads.json"

        with patch.object(Path, "parent", new_callable=lambda: property(lambda self: tmp_path)):
            # Simpler: just patch the manifest path lookup
            with patch("upload.Path") as MockPath:
                mock_manifest = tmp_path / ".chroma-uploads.json"
                MockPath.return_value.__truediv__ = lambda self, x: mock_manifest
                MockPath.__truediv__ = lambda self, x: mock_manifest

                plan = upload.get_upload_plan(client, "test-coll", git_meta, files, "source-code", str(git_repo))

        assert plan["action"] == "full"

    def test_same_commit_returns_skip(self, client, git_repo, tmp_path):
        git_meta = upload.get_git_metadata(str(git_repo))
        files = [str(git_repo / "main.py")]

        # Write a manifest entry matching current commit
        manifest_path = tmp_path / ".chroma-uploads.json"
        write_manifest(manifest_path, {
            "collection": "test-coll",
            "repo": git_meta["git_project_name"],
            "git_commit": git_meta["git_commit_hash"],
            "git_branch": git_meta["git_branch"],
            "files": 1,
            "chunks": 5,
        })

        # Patch manifest path
        orig_parent = Path(__file__).parent
        with patch("upload.Path.__new__", return_value=Path("dummy")):
            pass  # too complex to patch cleanly

        # Use monkeypatch approach instead
        import types

        original_fn = upload.get_upload_plan

        def patched_plan(client, collection_name, git_meta, all_files, store_type, input_path):
            # Temporarily replace manifest path
            import upload as mod
            orig = Path(mod.__file__).parent / ".chroma-uploads.json"
            try:
                # Copy our manifest to where upload.py looks
                import shutil
                shutil.copy2(manifest_path, orig)
                result = original_fn(client, collection_name, git_meta, all_files, store_type, input_path)
            finally:
                # Clean up
                if orig.exists():
                    orig.unlink()
            return result

        plan = patched_plan(client, "test-coll", git_meta, files, "source-code", str(git_repo))
        assert plan["action"] == "skip"

    def test_different_commit_returns_incremental(self, client, git_repo, tmp_path):
        git_meta = upload.get_git_metadata(str(git_repo))
        old_commit = git_meta["git_commit_hash"]

        # Make a new commit
        (git_repo / "new.py").write_text("z = 1\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "new file"], cwd=git_repo, capture_output=True)
        new_meta = upload.get_git_metadata(str(git_repo))
        files = [str(git_repo / "main.py"), str(git_repo / "utils.py"), str(git_repo / "new.py")]

        # Manifest has old commit
        manifest_path = tmp_path / ".chroma-uploads.json"
        write_manifest(manifest_path, {
            "collection": "test-coll",
            "repo": new_meta["git_project_name"],
            "git_commit": old_commit,
            "git_branch": new_meta["git_branch"],
            "files": 2,
            "chunks": 10,
        })

        # Copy manifest to where upload.py reads it
        real_manifest = Path(upload.__file__).parent / ".chroma-uploads.json"
        had_existing = real_manifest.exists()
        existing_content = real_manifest.read_text() if had_existing else None
        try:
            import shutil
            shutil.copy2(manifest_path, real_manifest)
            plan = upload.get_upload_plan(client, "test-coll", new_meta, files, "source-code", str(git_repo))
        finally:
            if had_existing:
                real_manifest.write_text(existing_content)
            elif real_manifest.exists():
                real_manifest.unlink()

        assert plan["action"] == "incremental"
        upload_names = [os.path.basename(f) for f in plan["upload"]]
        assert "new.py" in upload_names
        assert plan["delete"] == []  # no deletions, only an addition


# --- stash_if_dirty / stash_pop ---


class TestStash:
    def test_clean_repo_no_stash(self, git_repo):
        result = upload.stash_if_dirty(str(git_repo))
        assert result is None

    def test_dirty_repo_stashes_and_pops(self, git_repo):
        (git_repo / "dirty.py").write_text("dirty = True\n")
        assert (git_repo / "dirty.py").exists()

        stashed_root = upload.stash_if_dirty(str(git_repo))
        assert stashed_root is not None
        # Dirty file should be gone after stash
        assert not (git_repo / "dirty.py").exists()

        upload.stash_pop(stashed_root)
        # Dirty file should be back
        assert (git_repo / "dirty.py").exists()

    def test_not_a_git_repo(self, tmp_path):
        result = upload.stash_if_dirty(str(tmp_path))
        assert result is None


# --- End-to-end: full upload then incremental ---


class TestEndToEnd:
    def test_full_then_skip(self, client, collection, git_repo):
        """Upload a repo, then verify same commit is skipped."""
        git_meta = upload.get_git_metadata(str(git_repo))
        files = [str(git_repo / "main.py"), str(git_repo / "utils.py")]

        # First call: no manifest entry → full
        real_manifest = Path(upload.__file__).parent / ".chroma-uploads.json"
        had_existing = real_manifest.exists()
        existing_content = real_manifest.read_text() if had_existing else None
        try:
            # Ensure no manifest entry for this project
            temp_manifest = Path(upload.__file__).parent / ".chroma-uploads-backup.json"
            if had_existing:
                real_manifest.rename(temp_manifest)

            plan1 = upload.get_upload_plan(
                client, collection.name, git_meta, files, "source-code", str(git_repo)
            )
            assert plan1["action"] == "full"

            # Simulate successful upload by writing manifest
            write_manifest(real_manifest, {
                "collection": collection.name,
                "repo": git_meta["git_project_name"],
                "git_commit": git_meta["git_commit_hash"],
                "git_branch": git_meta["git_branch"],
                "files": 2,
                "chunks": 4,
            })

            # Second call: same commit → skip
            plan2 = upload.get_upload_plan(
                client, collection.name, git_meta, files, "source-code", str(git_repo)
            )
            assert plan2["action"] == "skip"

        finally:
            if real_manifest.exists():
                real_manifest.unlink()
            if temp_manifest.exists():
                temp_manifest.rename(real_manifest)

    def test_incremental_after_file_change(self, client, collection, git_repo):
        """Upload, change one file, verify incremental picks it up."""
        git_meta = upload.get_git_metadata(str(git_repo))
        old_commit = git_meta["git_commit_hash"]

        # Seed collection with chunks
        collection.add(
            ids=["main_001", "utils_001"],
            documents=["def hello(): print('hello')", "def add(a,b): return a+b"],
            metadatas=[
                {"file_path": str(git_repo / "main.py"), "git_project_name": "test-repo"},
                {"file_path": str(git_repo / "utils.py"), "git_project_name": "test-repo"},
            ],
        )

        # Modify one file and commit
        (git_repo / "main.py").write_text("def hello():\n    print('changed!')\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "change main"], cwd=git_repo, capture_output=True)
        new_meta = upload.get_git_metadata(str(git_repo))
        files = [str(git_repo / "main.py"), str(git_repo / "utils.py")]

        # Write manifest with old commit
        real_manifest = Path(upload.__file__).parent / ".chroma-uploads.json"
        had_existing = real_manifest.exists()
        existing_content = real_manifest.read_text() if had_existing else None
        temp_manifest = Path(upload.__file__).parent / ".chroma-uploads-backup.json"
        try:
            if had_existing:
                real_manifest.rename(temp_manifest)

            write_manifest(real_manifest, {
                "collection": collection.name,
                "repo": new_meta["git_project_name"],
                "git_commit": old_commit,
                "git_branch": new_meta["git_branch"],
                "files": 2,
                "chunks": 2,
            })

            plan = upload.get_upload_plan(
                client, collection.name, new_meta, files, "source-code", str(git_repo)
            )

            assert plan["action"] == "incremental"
            upload_names = [os.path.basename(f) for f in plan["upload"]]
            delete_names = [os.path.basename(f) for f in plan["delete"]]
            assert "main.py" in upload_names
            assert "main.py" in delete_names  # modified = delete old + upload new
            assert "utils.py" not in upload_names

            # Actually delete the old chunks
            deleted = upload.delete_file_chunks(collection, plan["delete"])
            assert deleted > 0
            # utils.py chunk should still be there
            assert collection.count() == 1

        finally:
            if real_manifest.exists():
                real_manifest.unlink()
            if temp_manifest.exists():
                temp_manifest.rename(real_manifest)

    def test_cross_branch_incremental(self, client, collection, git_repo):
        """Switch branches, verify incremental diff works across branches."""
        git_meta = upload.get_git_metadata(str(git_repo))
        main_commit = git_meta["git_commit_hash"]

        # Seed collection
        collection.add(
            ids=["main_001", "utils_001"],
            documents=["hello code", "utils code"],
            metadatas=[
                {"file_path": str(git_repo / "main.py"), "git_project_name": "test-repo"},
                {"file_path": str(git_repo / "utils.py"), "git_project_name": "test-repo"},
            ],
        )

        # Create feature branch with new file
        subprocess.run(["git", "checkout", "-b", "feature"], cwd=git_repo, capture_output=True)
        (git_repo / "feature.py").write_text("def feat(): pass\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "feature work"], cwd=git_repo, capture_output=True)
        feature_meta = upload.get_git_metadata(str(git_repo))
        files = [str(git_repo / "main.py"), str(git_repo / "utils.py"), str(git_repo / "feature.py")]

        real_manifest = Path(upload.__file__).parent / ".chroma-uploads.json"
        had_existing = real_manifest.exists()
        existing_content = real_manifest.read_text() if had_existing else None
        temp_manifest = Path(upload.__file__).parent / ".chroma-uploads-backup.json"
        try:
            if had_existing:
                real_manifest.rename(temp_manifest)

            # Manifest has main branch commit
            write_manifest(real_manifest, {
                "collection": collection.name,
                "repo": "test-repo",
                "git_commit": main_commit,
                "git_branch": "main",
                "files": 2,
                "chunks": 2,
            })

            plan = upload.get_upload_plan(
                client, collection.name, feature_meta, files, "source-code", str(git_repo)
            )

            assert plan["action"] == "incremental"
            upload_names = [os.path.basename(f) for f in plan["upload"]]
            assert "feature.py" in upload_names
            assert "main.py" not in upload_names  # unchanged across branches
            assert "utils.py" not in upload_names

        finally:
            if real_manifest.exists():
                real_manifest.unlink()
            if temp_manifest.exists():
                temp_manifest.rename(real_manifest)
