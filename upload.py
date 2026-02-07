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
from typing import List, Dict, Any, Tuple, Optional
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
        'markdown': ['.md', '.txt', '.rst', '.adoc', '.html', '.xml'],
        'json': ['.json', '.jsonl', '.geojson'],
        'txt': ['.txt'],
        'html': ['.html'],
        'xml': ['.xml'],
        'config': ['.toml', '.ini', '.conf', '.yaml', '.yml']
    }
    return extensions.get(store_type, [])

def find_git_projects(input_path: str, depth: Optional[int] = None) -> List[str]:
    """Find git project roots within input_path, optionally limited by depth."""
    root = Path(input_path).resolve()
    git_projects: List[str] = []

    for current_root, dirnames, _ in os.walk(root):
        current_path = Path(current_root)
        rel = current_path.relative_to(root)

        if depth is not None and len(rel.parts) > depth:
            dirnames[:] = []
            continue

        if '.git' in dirnames:
            git_projects.append(str(current_path))
            dirnames.remove('.git')

    return sorted(set(git_projects))

def find_files(input_path: str, store_type: str, limit: int = None, git_depth: Optional[int] = None) -> List[str]:
    """Find files matching store type, respecting .gitignore when possible."""
    extensions = get_file_extensions(store_type)
    files: List[str] = []
    abs_input = os.path.abspath(input_path)

    if os.path.isfile(abs_input):
        if any(abs_input.endswith(ext) for ext in extensions):
            return [abs_input]
        return []

    if store_type == 'source-code':
        git_projects = find_git_projects(abs_input, git_depth)
        if git_projects:
            for project_root in git_projects:
                try:
                    git_files = subprocess.check_output(
                        ['git', 'ls-files'],
                        cwd=project_root,
                        stderr=subprocess.PIPE,
                        text=True
                    ).strip().split('\n')
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

                for git_file in git_files:
                    if not git_file:
                        continue
                    if not any(git_file.endswith(ext) for ext in extensions):
                        continue
                    full_path = os.path.join(project_root, git_file)
                    if not full_path.startswith(abs_input):
                        continue
                    files.append(full_path)
                    if limit and len(files) >= limit:
                        return files

            print(f"  ✓ Using git ls-files ({len(files)} files, {len(git_projects)} git projects)")
            return files

    # Fallback: Use os.walk for non-git directories or non-source-code stores
    for root, _, filenames in os.walk(abs_input):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
                if limit and len(files) >= limit:
                    return files

    print(f"  ⚠ Using os.walk ({len(files)} files, .gitignore not respected)")
    return files

def get_model_defaults(model: str, store_type: str) -> Tuple[int, int]:
    """Return model-optimized chunk size and overlap."""
    if model == 'stella':
        chunk_size = 460
        chunk_overlap = 46
    elif model == 'modernbert':
        chunk_size = 920
        chunk_overlap = 92
    elif model == 'bge-large':
        chunk_size = 460
        chunk_overlap = 46
    else:
        chunk_size = 460
        chunk_overlap = 46

    if store_type == 'source-code':
        chunk_size = max(100, chunk_size - 60)
        chunk_overlap = max(0, chunk_overlap // 2)
    elif store_type == 'markdown':
        chunk_size = max(100, chunk_size - 30)

    return chunk_size, chunk_overlap

def normalize_chunk_settings(args) -> Tuple[int, int]:
    """Resolve chunk size/overlap values, supporting 'auto'."""
    if str(args.chunk_size).lower() == 'auto' or str(args.chunk_overlap).lower() == 'auto':
        default_size, default_overlap = get_model_defaults(args.embedding_model, args.store)
    else:
        default_size, default_overlap = get_model_defaults(args.embedding_model, args.store)

    if str(args.chunk_size).lower() == 'auto':
        chunk_size = default_size
    else:
        chunk_size = int(args.chunk_size)

    if str(args.chunk_overlap).lower() == 'auto':
        chunk_overlap = default_overlap
    else:
        chunk_overlap = int(args.chunk_overlap)

    return chunk_size, chunk_overlap

def create_client(args):
    """Create ChromaDB client based on args."""
    if args.data_path:
        return chromadb.PersistentClient(path=args.data_path)
    return chromadb.HttpClient(host=args.host, port=int(args.port))

def delete_project(client, collection_name: str, project_name: str) -> bool:
    """Delete all chunks for a given git project name from a collection."""
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return False

    batch_size = 1000
    offset = 0
    all_ids: List[str] = []

    while True:
        batch = collection.get(
            where={'git_project_name': project_name},
            include=[],
            limit=batch_size,
            offset=offset
        )
        if not batch.get('ids'):
            break
        all_ids.extend(batch['ids'])
        if len(batch['ids']) < batch_size:
            break
        offset += batch_size

    if not all_ids:
        try:
            batch = collection.get(include=['metadatas'], limit=10000)
            for doc_id, meta in zip(batch.get('ids', []), batch.get('metadatas', [])):
                if not meta:
                    continue
                root = meta.get('git_project_root')
                if root and project_name in root:
                    all_ids.append(doc_id)
        except Exception:
            pass

    if not all_ids:
        return False

    delete_batch = 100
    for i in range(0, len(all_ids), delete_batch):
        collection.delete(ids=all_ids[i:i + delete_batch])

    return True

def extract_pdf_text(
    pdf_path: str,
    ocr_enabled: bool,
    ocr_engine: str,
    ocr_language: str
) -> Tuple[str, str, Optional[float], bool]:
    """Extract text from PDF using PyMuPDF, with optional OCR fallback."""
    import fitz  # pymupdf

    text = ''
    extraction_method = 'pymupdf'
    ocr_confidence: Optional[float] = None
    is_image_pdf = False

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + '\n'
    doc.close()

    if text.strip():
        return text, extraction_method, ocr_confidence, is_image_pdf

    if not ocr_enabled:
        return text, extraction_method, ocr_confidence, is_image_pdf

    is_image_pdf = True
    extraction_method = f'ocr_{ocr_engine}'

    if ocr_engine not in ('tesseract', 'easyocr'):
        raise ValueError(f"Unsupported OCR engine: {ocr_engine}")

    try:
        if ocr_engine == 'tesseract':
            import pytesseract
            from PIL import Image
            import io

            pytesseract.get_tesseract_version()
            doc = fitz.open(pdf_path)
            ocr_text_parts = []
            confidence_scores = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                data = pytesseract.image_to_data(
                    image,
                    lang=ocr_language,
                    output_type=pytesseract.Output.DICT
                )
                page_text = pytesseract.image_to_string(image, lang=ocr_language)
                confidences = [int(conf) for conf in data.get('conf', []) if str(conf).isdigit() and int(conf) > 0]
                if confidences:
                    confidence_scores.extend(confidences)

                if page_text.strip():
                    ocr_text_parts.append(page_text)

            doc.close()
            text = '\n'.join(ocr_text_parts)
            if confidence_scores:
                ocr_confidence = sum(confidence_scores) / len(confidence_scores)

        elif ocr_engine == 'easyocr':
            import easyocr
            import numpy as np
            from PIL import Image
            import io

            doc = fitz.open(pdf_path)
            ocr_text_parts = []
            confidence_scores = []
            reader = easyocr.Reader([ocr_language])

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                img_array = np.array(image)

                results = reader.readtext(img_array, detail=1)
                page_text = ' '.join([result[1] for result in results])
                page_confidences = [result[2] * 100 for result in results]
                confidence_scores.extend(page_confidences)

                if page_text.strip():
                    ocr_text_parts.append(page_text)

            doc.close()
            text = '\n'.join(ocr_text_parts)
            if confidence_scores:
                ocr_confidence = sum(confidence_scores) / len(confidence_scores)

    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}") from e

    return text, extraction_method, ocr_confidence, is_image_pdf

def process_file(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    store_type: str,
    model_name: str,
    metadata_extra: Dict[str, Any],
    ocr_enabled: bool = True,
    ocr_engine: str = 'tesseract',
    ocr_language: str = 'eng'
) -> Dict[str, Any]:
    """
    Process a single file and return chunks (doesn't upload yet)
    Returns dict with ids, documents, metadatas for batch uploading
    """
    try:
        if store_type == 'pdf':
            content, extraction_method, ocr_confidence, is_image_pdf = extract_pdf_text(
                file_path, ocr_enabled, ocr_engine, ocr_language
            )
            if not content.strip():
                return {'ids': [], 'documents': [], 'metadatas': [], 'chunks': 0}

            chunks = chunk_utils.chunk_text_token_aware(
                content, chunk_size, chunk_overlap, model_name
            )
            chunk_metadata = None
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return {'ids': [], 'documents': [], 'metadatas': [], 'chunks': 0}

            # Chunk the content using appropriate strategy
            chunks, extraction_method, chunk_metadata = chunk_utils.chunk_text(
                content, file_path, store_type, chunk_size, model_name, chunk_overlap
            )
            ocr_confidence = None
            is_image_pdf = False

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
                'chunk_overlap_tokens': chunk_overlap,
                'text_extraction_method': extraction_method,
                'file_size': os.path.getsize(file_path),
                **git_meta,  # Add git metadata
                **metadata_extra
            }

            # Add markdown metadata if available
            if chunk_metadata and i < len(chunk_metadata):
                meta.update(chunk_metadata[i])

            if store_type == 'pdf':
                meta.update({
                    'is_image_pdf': is_image_pdf,
                    'ocr_enabled': ocr_enabled,
                    'ocr_engine': ocr_engine if is_image_pdf else '',
                    'ocr_language': ocr_language if is_image_pdf else '',
                    'ocr_confidence': ocr_confidence if ocr_confidence is not None else 0.0
                })

            elif store_type == 'source-code':
                language = chunk_utils.detect_language(file_path)
                if language:
                    meta['language'] = language

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
    args_tuple: (
        files, worker_id, total_workers, collection_name,
        host, port, data_path, chunk_size, chunk_overlap,
        store_type, model_name, metadata_base, batch_size,
        ocr_enabled, ocr_engine, ocr_language, dry_run
    )
    """
    (
        files, worker_id, total_workers, collection_name,
        host, port, data_path, chunk_size, chunk_overlap,
        store_type, model_name, metadata_base, batch_size,
        ocr_enabled, ocr_engine, ocr_language, dry_run
    ) = args_tuple

    # Each worker creates its own connection
    if data_path:
        client = chromadb.PersistentClient(path=data_path)
    else:
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
        file_result = process_file(
            file_path, chunk_size, chunk_overlap, store_type, model_name,
            metadata_base, ocr_enabled, ocr_engine, ocr_language
        )

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
            if not dry_run:
                upload_batch(collection, current_batch)
                batches_uploaded += 1
            current_batch = {'ids': [], 'documents': [], 'metadatas': []}

    # Upload remaining chunks
    if current_batch['ids']:
        if not dry_run:
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
    parser.add_argument('-i', '--input-path', help='Input path')
    parser.add_argument('--store', default='pdf', help='Store type (pdf, source-code, markdown)')
    parser.add_argument('-e', '--embedding-model', default='stella', help='Embedding model')
    parser.add_argument('--host', default='localhost', help='ChromaDB host')
    parser.add_argument('--port', default='9000', help='ChromaDB port')
    parser.add_argument('-d', '--data-path', default='', help='Data path for persistent client')
    parser.add_argument('-l', '--limit', type=int, help='Limit number of files')
    parser.add_argument('--chunk-size', default='auto', help='Chunk size in tokens (or auto)')
    parser.add_argument('--chunk-overlap', default='auto', help='Chunk overlap in tokens (or auto)')
    parser.add_argument('--delete-collection', action='store_true', help='Delete collection first')
    parser.add_argument('--delete-project', default='', help='Delete specific git project from collection')
    parser.add_argument('--delete-failed-project', action='store_true', help='Auto-delete project if any upload fails')
    parser.add_argument('--force', action='store_true', help='Force upload even if git commit unchanged')
    parser.add_argument('--batch-size', type=int, default=50, help='Upload batch size (chunks)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1, use 0 for auto=CPU cores)')
    parser.add_argument('--dry-run', action='store_true', help='Preview chunking without uploading')
    parser.add_argument('--disable-ocr', action='store_true', help='Disable OCR for image PDFs')
    parser.add_argument('--ocr-engine', default='tesseract', help='OCR engine: tesseract or easyocr')
    parser.add_argument('--ocr-language', default='eng', help='OCR language code')
    parser.add_argument('--depth', type=int, help='Git project search depth (source-code store)')

    args = parser.parse_args()

    valid_models = {'stella', 'modernbert', 'bge-large', 'default'}
    if args.embedding_model not in valid_models:
        print(f"❌ Invalid embedding model: {args.embedding_model}")
        print("Valid options: stella, modernbert, bge-large, default")
        return

    valid_stores = {'pdf', 'source-code', 'markdown'}
    if args.store not in valid_stores:
        print(f"❌ Invalid store type: {args.store}")
        print("Valid options: pdf, source-code, markdown")
        return

    if args.ocr_engine not in ('tesseract', 'easyocr'):
        print(f"❌ Invalid OCR engine: {args.ocr_engine}")
        print("Valid options: tesseract, easyocr")
        return

    if args.depth is not None and args.depth < 1:
        print(f"❌ Invalid depth value: {args.depth}")
        print("Depth must be a positive integer (1 or higher)")
        return

    if args.store != 'pdf' and not args.disable_ocr:
        args.disable_ocr = True

    # Auto-detect workers
    if args.workers == 0:
        args.workers = max(1, cpu_count() // 2)  # Use half of CPU cores

    print()
    print("=" * 50)
    print("Fast Upload to ChromaDB (Parallel)")
    print("=" * 50)
    print(f"Collection: {args.collection}")
    if args.input_path:
        print(f"Input: {args.input_path}")
    print(f"Store: {args.store}")
    print(f"Model: {args.embedding_model}")
    if args.data_path:
        print(f"Client: persistent ({args.data_path})")
    else:
        print(f"Client: http ({args.host}:{args.port})")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size} chunks")
    print()

    # Connect to ChromaDB (persistent connection!)
    print("⏳ Connecting to ChromaDB...")
    start_time = time.time()
    client = create_client(args)
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

    if args.delete_project:
        deleted = delete_project(client, args.collection, args.delete_project)
        if deleted:
            print(f"✓ Deleted project '{args.delete_project}' from collection '{args.collection}'")
        else:
            print(f"⚠ Project '{args.delete_project}' not found in collection '{args.collection}'")
        return

    if not args.input_path:
        print("❌ Input path is required unless using --delete-project")
        return

    # Find files first (needed for completeness check)
    print("🔍 Finding files...")
    files = find_files(args.input_path, args.store, args.limit, args.depth)
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

    chunk_size, chunk_overlap = normalize_chunk_settings(args)
    print(f"Chunk size: {chunk_size} tokens")
    print(f"Chunk overlap: {chunk_overlap} tokens")
    print()

    if args.dry_run:
        print("Dry run: chunking without upload")
    # Process files
    print(f"📤 Processing files with {args.workers} workers...")
    start_time = time.time()

    metadata_base = {
        'embedding_model': args.embedding_model,
        'store_type': args.store,
        'upload_date': datetime.utcnow().isoformat() + 'Z',
        'storage': args.data_path if args.data_path else f"{args.host}:{args.port}",
        'is_new_upload': True
    }

    if args.workers == 1:
        # Single-threaded mode (original batching code)
        total_chunks = 0
        processed = 0
        failed = 0
        batches_uploaded = 0

        current_batch = {'ids': [], 'documents': [], 'metadatas': []}

        for i, file_path in enumerate(files, 1):
            file_result = process_file(
                file_path, chunk_size, chunk_overlap, args.store, args.embedding_model,
                metadata_base, not args.disable_ocr, args.ocr_engine, args.ocr_language
            )

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
                if not args.dry_run:
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
            if not args.dry_run:
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
                args.host, args.port, args.data_path, chunk_size, chunk_overlap,
                args.store, args.embedding_model, metadata_base, args.batch_size,
                not args.disable_ocr, args.ocr_engine, args.ocr_language, args.dry_run
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
    avg_chunk_size = chunk_size  # Use configured chunk size

    print()
    if args.dry_run:
        print("✅ Dry run complete!")
    else:
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
    if processed > 0 and not args.dry_run:
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

    if failed > 0 and args.delete_failed_project and not args.dry_run:
        input_git_meta = get_git_metadata(args.input_path)
        project_name = input_git_meta.get('git_project_name')
        if project_name:
            print(f"⚠ Upload failures detected; deleting project '{project_name}'")
            delete_project(client, args.collection, project_name)

    # Clear progress file after upload completes
    clear_progress()

if __name__ == '__main__':
    # Required for multiprocessing on macOS
    from multiprocessing import freeze_support
    freeze_support()
    main()
