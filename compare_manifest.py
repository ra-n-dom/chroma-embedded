#!/usr/bin/env python3
"""
Compare filesystem repos to manifest - find what's not uploaded yet
"""

import json
from pathlib import Path
import sys

def get_git_repos(projects_dir):
    """Find all git repos in a directory"""
    projects_path = Path(projects_dir).expanduser()
    repos = []

    for item in projects_path.iterdir():
        if item.is_dir() and (item / '.git').exists():
            repos.append(item.name)

    return sorted(repos)

def load_manifest(manifest_path):
    """Load repos from manifest"""
    manifest_file = Path(manifest_path)

    if not manifest_file.exists():
        return {}

    repos = {}
    with open(manifest_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            repo_name = entry['repo']
            repos[repo_name] = entry

    return repos

def compare(projects_dir, manifest_path='.chroma-uploads.json'):
    """Compare filesystem to manifest"""
    print()
    print("=" * 60)
    print("Comparing filesystem to upload manifest")
    print("=" * 60)
    print()

    # Get repos from filesystem
    fs_repos = set(get_git_repos(projects_dir))
    print(f"Found {len(fs_repos)} git repositories in {projects_dir}")

    # Get repos from manifest
    manifest_repos = load_manifest(manifest_path)
    print(f"Found {len(manifest_repos)} repositories in manifest")
    print()

    # Find missing (not uploaded)
    missing = fs_repos - set(manifest_repos.keys())

    if missing:
        print(f"⚠ NOT UPLOADED ({len(missing)} repos):")
        print("-" * 60)
        for repo in sorted(missing):
            repo_path = Path(projects_dir).expanduser() / repo
            try:
                # Count files quickly
                file_count = sum(1 for _ in repo_path.rglob("*") if _.is_file() and _.suffix in ['.py', '.java', '.js', '.ts', '.rb', '.go'])
                print(f"  {repo} ({file_count} source files)")
            except:
                print(f"  {repo}")
        print()
    else:
        print("✓ All repos are uploaded!")
        print()

    # Find in manifest but not on disk (stale/deleted)
    stale = set(manifest_repos.keys()) - fs_repos

    if stale:
        print(f"⚠ IN MANIFEST BUT NOT ON DISK ({len(stale)} repos):")
        print("-" * 60)
        for repo in sorted(stale):
            entry = manifest_repos[repo]
            print(f"  {repo} (uploaded {entry.get('upload_date', 'unknown')[:10]}, {entry['chunks']} chunks)")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print(f"  On disk: {len(fs_repos)}")
    print(f"  Uploaded: {len(manifest_repos)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Stale: {len(stale)}")
    print()

    if missing:
        print("To upload missing repos:")
        print(f"  python3 fast_upload.py -c projects -i ~/projects/REPO_NAME --store source-code -e stella")
        print()
        print("Or bulk upload all missing:")
        missing_file = Path('missing_repos.txt')
        with open(missing_file, 'w') as f:
            for repo in sorted(missing):
                f.write(f"{Path(projects_dir).expanduser() / repo}\n")
        print(f"  Created: {missing_file}")
        print(f"  Use with: cat {missing_file} | while read path; do")
        print(f"    python3 fast_upload.py -c projects -i \"$path\" --store source-code -e stella")
        print(f"  done")

if __name__ == '__main__':
    projects_dir = sys.argv[1] if len(sys.argv) > 1 else '~/projects'
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else '.chroma-uploads.json'
    compare(projects_dir, manifest_path)
