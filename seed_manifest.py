#!/usr/bin/env python3
"""
Seed .chroma-uploads.json manifest from existing ChromaDB collections
Extracts all uploaded repos and creates audit trail
"""

import chromadb
import json
from pathlib import Path
from collections import defaultdict

def seed_manifest(host='localhost', port='9000'):
    """Generate manifest from current ChromaDB state"""

    client = chromadb.HttpClient(host=host, port=int(port))
    collections = client.list_collections()

    print(f"Found {len(collections)} collections")
    print()

    manifest_file = Path(__file__).parent / '.chroma-uploads.json'

    # Clear existing manifest
    if manifest_file.exists():
        backup = Path(str(manifest_file) + '.backup')
        manifest_file.rename(backup)
        print(f"✓ Backed up existing manifest to {backup.name}")

    total_entries = 0

    for collection_obj in collections:
        collection_name = collection_obj.name
        print(f"Processing collection: {collection_name}")

        try:
            collection = client.get_collection(collection_name)

            # Get all documents with git metadata
            results = collection.get(
                limit=100000,
                include=['metadatas']
            )

            # Group by repository
            repos = defaultdict(lambda: {
                'chunks': 0,
                'files': set(),
                'commits': set(),
                'branches': set(),
                'remote_urls': set(),
                'upload_dates': [],
                'embedding_models': set(),
            })

            for metadata in results['metadatas']:
                repo_name = metadata.get('git_project_name')
                if not repo_name:
                    continue

                repos[repo_name]['chunks'] += 1

                if metadata.get('file_path'):
                    repos[repo_name]['files'].add(metadata['file_path'])
                if metadata.get('git_commit_hash'):
                    repos[repo_name]['commits'].add(metadata['git_commit_hash'])
                if metadata.get('git_branch'):
                    repos[repo_name]['branches'].add(metadata['git_branch'])
                if metadata.get('git_remote_url'):
                    repos[repo_name]['remote_urls'].add(metadata['git_remote_url'])
                if metadata.get('upload_date'):
                    repos[repo_name]['upload_dates'].append(metadata['upload_date'])
                if metadata.get('embedding_model'):
                    repos[repo_name]['embedding_models'].add(metadata['embedding_model'])

            # Write to JSONL
            for repo_name, stats in sorted(repos.items()):
                # Use most recent upload date
                upload_date = max(stats['upload_dates']) if stats['upload_dates'] else None

                entry = {
                    'collection': collection_name,
                    'repo': repo_name,
                    'git_commit': list(stats['commits'])[0] if stats['commits'] else None,
                    'git_branch': list(stats['branches'])[0] if stats['branches'] else None,
                    'git_remote_url': list(stats['remote_urls'])[0] if stats['remote_urls'] else None,
                    'upload_date': upload_date,
                    'files': len(stats['files']),
                    'chunks': stats['chunks'],
                    'embedding_models': list(stats['embedding_models']),
                }

                with open(manifest_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')

                total_entries += 1
                print(f"  • {repo_name}: {stats['chunks']} chunks, {len(stats['files'])} files")

        except Exception as e:
            print(f"  Error processing {collection_name}: {e}")

        print()

    print(f"✓ Manifest created: {manifest_file}")
    print(f"Total entries: {total_entries}")
    print()

    print("Example usage:")
    print(f"  # View all uploads")
    print(f"  cat {manifest_file}")
    print()
    print(f"  # Find specific repo")
    print(f"  grep 'chroma-embedded' {manifest_file}")
    print()
    print(f"  # Count repos per collection")
    print(f"  jq -r '.collection' {manifest_file} | sort | uniq -c")

if __name__ == '__main__':
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    port = sys.argv[2] if len(sys.argv) > 2 else '9000'

    seed_manifest(host, port)
