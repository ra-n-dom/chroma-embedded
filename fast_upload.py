#!/usr/bin/env python3
"""
Fast upload to ChromaDB - Pure Python version
Maintains persistent connection for massive speedup over upload.sh
Usage: python3 fast_upload.py --collection MyCollection --input /path/to/files --store source-code
"""

import argparse
import chromadb
import os
import time
import subprocess
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count
import sys
from datetime import datetime
import chunk_utils

def get_git_metadata(file_path: str) -> Dict[str, str]:
    """Extract git metadata for a file or directory"""
    try:
        # Handle both files and directories
        if os.path.isfile(file_path):
            work_dir = os.path.dirname(os.path.abspath(file_path))
        else:
            work_dir = os.path.abspath(file_path)

        # Find git root
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=work_dir,
            stderr=subprocess.PIPE,
            text=True
        ).strip()

        # Get git metadata
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=git_root,
            stderr=subprocess.PIPE,
            text=True
        ).strip()

        try:
            remote_url = subprocess.check_output(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=git_root,
                stderr=subprocess.PIPE,
                text=True
            ).strip()
        except:
            remote_url = 'unknown'

        branch = subprocess.check_output(
            ['git', 'branch', '--show-current'],
            cwd=git_root,
            stderr=subprocess.PIPE,
            text=True
        ).strip()

        project_name = os.path.basename(git_root)

        return {
            'git_project_root': git_root,
            'git_project_name': project_name,
            'git_commit_hash': commit_hash,
            'git_remote_url': remote_url,
            'git_branch': branch,
        }

    except Exception as e:
        # Not a git repo or git command failed
        return {}

def get_file_extensions(store_type: str) -> List[str]:
    """Get file extensions based on store type"""
    extensions = {
        'source-code': [
            '.py', '.java', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs',
            '.cpp', '.c', '.cs', '.php', '.rb', '.kt', '.scala', '.swift',
            '.hcl', '.nomad', '.tf', '.sh', '.bash'  # Config/infra files
        ],
        'pdf': ['.pdf'],
        'markdown': ['.md'],
        'json': ['.json', '.jsonl', '.geojson'],
        'txt': ['.txt'],
        'html': ['.html'],
        'xml': ['.xml'],
        'config': ['.toml', '.ini', '.conf', '.yaml', '.yml']
    }
    return extensions.get(store_type, [])

def find_files(input_path: str, store_type: str, limit: int = None) -> List[str]:
    """Find files matching store type, respecting .gitignore if in git repo"""
    extensions = get_file_extensions(store_type)
    files = []

    # Check if this is a git repository
    try:
        abs_input = os.path.abspath(input_path)
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=abs_input if os.path.isdir(abs_input) else os.path.dirname(abs_input),
            stderr=subprocess.PIPE,
            text=True
        ).strip()

        # Use git ls-files to respect .gitignore
        git_files = subprocess.check_output(
            ['git', 'ls-files'],
            cwd=git_root,
            stderr=subprocess.PIPE,
            text=True
        ).strip().split('\n')

        # Filter by extensions and path
        for git_file in git_files:
            full_path = os.path.join(git_root, git_file)

            # Check if file is within input_path
            if not full_path.startswith(abs_input):
                continue

            # Check if file matches store type
            if any(git_file.endswith(ext) for ext in extensions):
                files.append(full_path)
                if limit and len(files) >= limit:
                    return files

        print(f"  ✓ Using git ls-files ({len(files)} files, respecting .gitignore)")
        return files

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo or git not available, fall back to os.walk
        pass

    # Fallback: Use os.walk for non-git directories
    for root, dirs, filenames in os.walk(input_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
                if limit and len(files) >= limit:
                    return files

    print(f"  ⚠ Not a git repo, using os.walk ({len(files)} files, .gitignore not respected)")
    return files

def process_file(
    file_path: str,
    chunk_size: int,
    store_type: str,
    model_name: str,
    metadata_extra: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single file and return chunks (doesn't upload yet)
    Returns dict with ids, documents, metadatas for batch uploading
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip():
            return {'ids': [], 'documents': [], 'metadatas': [], 'chunks': 0}

        # Chunk the content using appropriate strategy
        chunks, extraction_method, chunk_metadata = chunk_utils.chunk_text(
            content, file_path, store_type, chunk_size, model_name
        )

        # Generate unique IDs using hash of full path
        import hashlib
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
        ids = [f"{path_hash}_chunk_{i:03d}" for i in range(len(chunks))]

        # Extract git metadata
        git_meta = get_git_metadata(file_path)

        metadatas = []
        for i in range(len(chunks)):
            meta = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'chunk_index': i,
                'chunk_count': len(chunks),
                'chunk_size_tokens': chunk_size,
                'text_extraction_method': extraction_method,
                **git_meta,  # Add git metadata
                **metadata_extra
            }

            # Add markdown metadata if available
            if chunk_metadata and i < len(chunk_metadata):
                meta.update(chunk_metadata[i])

            metadatas.append(meta)

        return {
            'ids': ids,
            'documents': chunks,
            'metadatas': metadatas,
            'chunks': len(chunks),
            'file': file_path
        }

    except Exception as e:
        print(f"  ⚠ Error processing {file_path}: {e}")
        return {'ids': [], 'documents': [], 'metadatas': [], 'chunks': 0, 'error': str(e)}

def upload_batch(collection, batch_data: Dict[str, list]) -> int:
    """Upload a batch of chunks to ChromaDB"""
    if not batch_data['ids']:
        return 0

    collection.add(
        ids=batch_data['ids'],
        documents=batch_data['documents'],
        metadatas=batch_data['metadatas']
    )
    return len(batch_data['ids'])

def update_progress(progress_data: Dict[str, Any]):
    """Update live progress file during upload"""
    progress_path = Path(__file__).parent / '.chroma-upload-progress.json'
    try:
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        print(f"  ⚠ Failed to update progress: {e}")

def clear_progress():
    """Clear progress file when upload completes"""
    progress_path = Path(__file__).parent / '.chroma-upload-progress.json'
    try:
        if progress_path.exists():
            progress_path.unlink()
    except Exception as e:
        print(f"  ⚠ Failed to clear progress: {e}")

def append_to_manifest(
    input_path: str,
    collection: str,
    processed: int,
    total_chunks: int,
    git_meta: Dict[str, str],
    performance_metrics: Dict[str, Any] = None
):
    """Append upload record to manifest"""
    manifest_path = Path(__file__).parent / '.chroma-uploads.json'

    # Only append if this was a git repo
    if not git_meta.get('git_project_name'):
        print(f"  ⚠ No git metadata found for {input_path} - skipping manifest update")
        return

    entry = {
        'collection': collection,
        'repo': git_meta.get('git_project_name'),
        'git_commit': git_meta.get('git_commit_hash'),
        'git_branch': git_meta.get('git_branch'),
        'git_remote_url': git_meta.get('git_remote_url'),
        'upload_date': datetime.utcnow().isoformat() + 'Z',
        'files': processed,
        'chunks': total_chunks,
        'input_path': input_path,
    }

    # Add performance metrics if provided
    if performance_metrics:
        entry.update({
            'upload_duration_seconds': performance_metrics.get('duration_seconds'),
            'chunks_per_second': performance_metrics.get('chunks_per_second'),
            'files_per_minute': performance_metrics.get('files_per_minute'),
            'avg_chunk_size_tokens': performance_metrics.get('avg_chunk_size_tokens'),
            'store_type': performance_metrics.get('store_type'),
            'file_types': performance_metrics.get('file_types', {})
        })

    try:
        with open(manifest_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        print(f"  ✓ Updated manifest: {manifest_path}")
    except Exception as e:
        print(f"  ⚠ Failed to update manifest: {e}")

def check_if_upload_needed(
    client,
    collection_name: str,
    git_meta: Dict[str, str],
    expected_file_count: int
) -> bool:
    """
    Check if upload is needed based on git commit comparison AND manifest verification.
    Returns True if upload needed, False if can skip.
    """
    project_name = git_meta.get('git_project_name')
    current_commit = git_meta.get('git_commit_hash')

    if not project_name or not current_commit:
        # No git metadata, can't do incremental check
        return True

    # Check manifest for last successful upload
    manifest_path = Path(__file__).parent / '.chroma-uploads.json'

    try:
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                lines = f.readlines()

            # Find most recent entry for this project
            for line in reversed(lines):
                try:
                    entry = json.loads(line.strip())
                    if entry.get('repo') == project_name and entry.get('collection') == collection_name:
                        manifest_commit = entry.get('git_commit')
                        manifest_files = entry.get('files', 0)
                        manifest_chunks = entry.get('chunks', 0)

                        if manifest_commit == current_commit:
                            # Same commit - check if upload completed successfully
                            if manifest_files == expected_file_count:
                                # Manifest shows complete upload
                                print(f"  ✓ Project '{project_name}' unchanged (commit {current_commit[:8]})")
                                print(f"  ✓ Last upload: {manifest_files} files → {manifest_chunks} chunks")
                                print(f"  ⚠ Skipping upload (use --force to override)")
                                return False
                            else:
                                # File count mismatch - incomplete upload
                                print(f"  ⚠ Project '{project_name}' appears incomplete:")
                                print(f"    Commit: {current_commit[:8]} (unchanged)")
                                print(f"    Expected: {expected_file_count} files")
                                print(f"    Last upload: {manifest_files} files → {manifest_chunks} chunks")
                                print(f"    Will re-upload to complete")
                                return True
                        else:
                            # Commit changed
                            print(f"  ⚠ Project '{project_name}' changed:")
                            print(f"    Old: {manifest_commit[:8] if manifest_commit else 'unknown'}")
                            print(f"    New: {current_commit[:8]}")
                            print(f"    Will upload changed files")
                            return True

                except (json.JSONDecodeError, KeyError):
                    continue

    except Exception as e:
        print(f"  ⚠ Could not read manifest: {e}")

    # No manifest entry found - proceed with upload
    print(f"  ⚠ No previous upload found for '{project_name}'")
    return True

def process_file_batch(args_tuple):
    """
    Worker function for parallel processing
    args_tuple: (files, worker_id, total_workers, collection_name, host, port, chunk_size, store_type, model_name, metadata_base, batch_size)
    """
    files, worker_id, total_workers, collection_name, host, port, chunk_size, store_type, model_name, metadata_base, batch_size = args_tuple

    # Each worker creates its own connection
    client = chromadb.HttpClient(host=host, port=int(port))
    collection = client.get_or_create_collection(name=collection_name)

    total_chunks = 0
    processed = 0
    failed = 0
    batches_uploaded = 0

    current_batch = {
        'ids': [],
        'documents': [],
        'metadatas': []
    }

    for i, file_path in enumerate(files):
        file_result = process_file(file_path, chunk_size, store_type, model_name, metadata_base)

        if file_result.get('error'):
            failed += 1
            continue

        chunk_count = file_result['chunks']
        if chunk_count == 0:
            continue

        # Add to batch
        current_batch['ids'].extend(file_result['ids'])
        current_batch['documents'].extend(file_result['documents'])
        current_batch['metadatas'].extend(file_result['metadatas'])

        total_chunks += chunk_count
        processed += 1

        # Upload batch when it reaches batch_size
        if len(current_batch['ids']) >= batch_size:
            upload_batch(collection, current_batch)
            batches_uploaded += 1
            current_batch = {'ids': [], 'documents': [], 'metadatas': []}

    # Upload remaining chunks
    if current_batch['ids']:
        upload_batch(collection, current_batch)
        batches_uploaded += 1

    return {
        'worker_id': worker_id,
        'processed': processed,
        'failed': failed,
        'total_chunks': total_chunks,
        'batches': batches_uploaded
    }

def main():
    parser = argparse.ArgumentParser(description='Fast upload to ChromaDB')
    parser.add_argument('-c', '--collection', required=True, help='Collection name')
    parser.add_argument('-i', '--input-path', required=True, help='Input path')
    parser.add_argument('--store', default='source-code', help='Store type')
    parser.add_argument('-e', '--embedding-model', default='stella', help='Embedding model')
    parser.add_argument('--host', default='localhost', help='ChromaDB host')
    parser.add_argument('--port', default='9000', help='ChromaDB port')
    parser.add_argument('-l', '--limit', type=int, help='Limit number of files')
    parser.add_argument('--chunk-size', type=int, default=400, help='Chunk size in tokens')
    parser.add_argument('--delete-collection', action='store_true', help='Delete collection first')
    parser.add_argument('--force', action='store_true', help='Force upload even if git commit unchanged')
    parser.add_argument('--batch-size', type=int, default=50, help='Upload batch size (chunks)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1, use 0 for auto=CPU cores)')

    args = parser.parse_args()

    # Auto-detect workers
    if args.workers == 0:
        args.workers = max(1, cpu_count() // 2)  # Use half of CPU cores

    print()
    print("=" * 50)
    print("Fast Upload to ChromaDB (Parallel)")
    print("=" * 50)
    print(f"Collection: {args.collection}")
    print(f"Input: {args.input_path}")
    print(f"Store: {args.store}")
    print(f"Model: {args.embedding_model}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size} chunks")
    print()

    # Connect to ChromaDB (persistent connection!)
    print("⏳ Connecting to ChromaDB...")
    start_time = time.time()
    client = chromadb.HttpClient(host=args.host, port=int(args.port))
    connect_time = time.time() - start_time
    print(f"✓ Connected in {connect_time:.2f}s")
    print()

    # Get or create collection
    if args.delete_collection:
        try:
            client.delete_collection(args.collection)
            print(f"✓ Deleted existing collection")
        except:
            pass

    collection = client.get_or_create_collection(name=args.collection)
    print(f"✓ Using collection: {args.collection}")
    print()

    # Find files first (needed for completeness check)
    print("🔍 Finding files...")
    files = find_files(args.input_path, args.store, args.limit)
    print(f"Found {len(files)} files")
    print()

    # Check if incremental upload can skip this project
    if not args.delete_collection and not args.force:
        input_git_meta = get_git_metadata(args.input_path)
        if not check_if_upload_needed(client, args.collection, input_git_meta, len(files)):
            print()
            print("✓ Upload skipped (no changes)")
            return

    if len(files) == 0:
        print("No files to upload")
        return

    # Process files
    print(f"📤 Processing files with {args.workers} workers...")
    start_time = time.time()

    metadata_base = {
        'embedding_model': args.embedding_model,
        'store_type': args.store,
    }

    if args.workers == 1:
        # Single-threaded mode (original batching code)
        total_chunks = 0
        processed = 0
        failed = 0
        batches_uploaded = 0

        current_batch = {'ids': [], 'documents': [], 'metadatas': []}

        for i, file_path in enumerate(files, 1):
            file_result = process_file(file_path, args.chunk_size, args.store, args.embedding_model, metadata_base)

            if file_result.get('error'):
                failed += 1
                print(f"✗ [{i}/{len(files)}] {os.path.basename(file_path)} - FAILED")
                continue

            chunk_count = file_result['chunks']
            if chunk_count == 0:
                continue

            current_batch['ids'].extend(file_result['ids'])
            current_batch['documents'].extend(file_result['documents'])
            current_batch['metadatas'].extend(file_result['metadatas'])

            total_chunks += chunk_count
            processed += 1

            print(f"✓ [{i}/{len(files)}] {os.path.basename(file_path)} ({chunk_count} chunks)")

            if len(current_batch['ids']) >= args.batch_size:
                upload_batch(collection, current_batch)
                batches_uploaded += 1
                current_batch = {'ids': [], 'documents': [], 'metadatas': []}

            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed * 60
                eta_seconds = (len(files) - i) / (i / elapsed) if elapsed > 0 else 0
                eta_min = int(eta_seconds / 60)
                print(f"  Progress: {i}/{len(files)} ({rate:.0f} files/min, ETA: {eta_min}min)")

                # Update progress file
                input_git_meta = get_git_metadata(args.input_path)
                update_progress({
                    'status': 'uploading',
                    'repo': input_git_meta.get('git_project_name', 'unknown'),
                    'collection': args.collection,
                    'current_file': i,
                    'total_files': len(files),
                    'processed': processed,
                    'failed': failed,
                    'chunks': total_chunks,
                    'elapsed_seconds': elapsed,
                    'files_per_minute': rate,
                    'eta_seconds': eta_seconds,
                    'store_type': args.store,
                    'start_time': datetime.fromtimestamp(start_time).isoformat() + 'Z'
                })

        if current_batch['ids']:
            upload_batch(collection, current_batch)
            batches_uploaded += 1

    else:
        # Multi-threaded mode with parallel workers
        print(f"Dividing {len(files)} files among {args.workers} workers...")

        # Divide files among workers
        files_per_worker = len(files) // args.workers
        worker_args = []

        for worker_id in range(args.workers):
            start_idx = worker_id * files_per_worker
            if worker_id == args.workers - 1:
                # Last worker gets remaining files
                worker_files = files[start_idx:]
            else:
                worker_files = files[start_idx:start_idx + files_per_worker]

            worker_args.append((
                worker_files, worker_id, args.workers, args.collection,
                args.host, args.port, args.chunk_size, args.store,
                args.embedding_model, metadata_base, args.batch_size
            ))

        # Run workers in parallel
        with Pool(processes=args.workers) as pool:
            results = pool.map(process_file_batch, worker_args)

        # Aggregate results
        total_chunks = sum(r['total_chunks'] for r in results)
        processed = sum(r['processed'] for r in results)
        failed = sum(r['failed'] for r in results)
        batches_uploaded = sum(r['batches'] for r in results)

        print(f"\n✓ All workers completed")
        for r in results:
            print(f"  Worker {r['worker_id']}: {r['processed']} files, {r['total_chunks']} chunks, {r['batches']} batches")

    # Summary
    elapsed = time.time() - start_time
    rate = len(files) / elapsed * 60
    chunks_per_batch = total_chunks / batches_uploaded if batches_uploaded > 0 else 0
    chunks_per_second = total_chunks / elapsed if elapsed > 0 else 0
    avg_chunk_size = args.chunk_size  # Use configured chunk size

    print()
    print("✅ Upload complete!")
    print(f"Files: {processed} succeeded, {failed} failed")
    print(f"Total chunks: {total_chunks}")
    print(f"Batches: {batches_uploaded} ({chunks_per_batch:.0f} chunks/batch avg)")
    print(f"Time: {elapsed:.1f}s ({int(elapsed/60)}m {int(elapsed%60)}s)")
    print(f"Speed: {rate:.0f} files/min, {chunks_per_second:.1f} chunks/sec")

    # Calculate file type distribution
    file_types = {}
    for file_path in files:
        ext = Path(file_path).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    # Determine if AST-aware chunking was used
    # AST-aware applies to source-code store type with supported languages
    ast_aware_extensions = {'.py', '.java', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs',
                           '.cpp', '.c', '.cs', '.kt', '.scala', '.swift'}
    uses_ast = args.store == 'source-code' and any(ext in ast_aware_extensions for ext in file_types.keys())

    # Update manifest with performance metrics
    if processed > 0:
        input_git_meta = get_git_metadata(args.input_path)
        performance_metrics = {
            'duration_seconds': round(elapsed, 2),
            'chunks_per_second': round(chunks_per_second, 2),
            'files_per_minute': round(rate, 2),
            'avg_chunk_size_tokens': avg_chunk_size,
            'store_type': args.store,
            'file_types': file_types,
            'ast_aware': uses_ast
        }
        append_to_manifest(args.input_path, args.collection, processed, total_chunks, input_git_meta, performance_metrics)

    # Clear progress file after upload completes
    clear_progress()

if __name__ == '__main__':
    # Required for multiprocessing on macOS
    from multiprocessing import freeze_support
    freeze_support()
    main()
