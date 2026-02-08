#!/usr/bin/env python3
"""
Chunking utilities for ChromaDB uploads
Supports AST-aware, token-aware, and markdown heading-aware chunking
"""

import os
from typing import List, Dict, Tuple, Optional

def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension"""
    language_map = {
        '.py': 'python',
        '.java': 'java',
        '.js': 'typescript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'typescript',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.php': 'php',
        '.rb': 'ruby',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.swift': 'swift',
    }

    file_ext = os.path.splitext(file_path)[1].lower()
    return language_map.get(file_ext)

def chunk_text_token_aware(
    text: str,
    chunk_size_tokens: int = 460,
    overlap_tokens: int = 46,
    model_name: str = 'stella'
) -> List[str]:
    """
    Chunk text based on model-specific character-to-token ratios.
    Uses researched ratios for each embedding model.
    """

    # Model-specific character-to-token ratios based on research
    model_char_ratios = {
        'stella': 3.2,        # BERT-based, WordPiece tokenizer
        'modernbert': 3.4,    # Modern BERT with improved tokenization
        'bge-large': 3.3,     # BERT-based, similar to Stella
        'default': 3.5        # all-MiniLM-L6-v2, sentence-transformers default
    }

    chars_per_token = model_char_ratios.get(model_name, 3.5)
    chunk_size_chars = int(chunk_size_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size_chars
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap_chars
        if start >= len(text):
            break

    return chunks

def chunk_source_code_ast(
    text: str,
    file_path: str,
    chunk_size_tokens: int = 400,
    model_name: str = 'stella',
    verbose: bool = False
) -> Tuple[List[str], str]:
    """
    Chunk source code using AST-aware chunking.
    Falls back to token-aware chunking if AST chunking fails.

    Returns:
        Tuple of (chunks, extraction_method)
    """
    # Languages confirmed to work with ASTChunk (tested)
    AST_SUPPORTED = ('python', 'java', 'typescript')

    # Languages confirmed to NOT work (tested - skip AST attempt for performance)
    AST_UNSUPPORTED = ('ruby', 'go', 'kotlin', 'c', 'cpp', 'csharp', 'php', 'rust', 'scala', 'swift')

    language = detect_language(file_path)
    if not language:
        # Not a recognized source code file, use token-aware
        chunks = chunk_text_token_aware(text, chunk_size_tokens, 0, model_name)
        return chunks, 'token_aware'

    # Skip AST for known unsupported languages (performance optimization)
    if language in AST_UNSUPPORTED:
        chunks = chunk_text_token_aware(text, chunk_size_tokens, 0, model_name)
        return chunks, 'token_aware'

    try:
        from astchunk import ASTChunkBuilder

        # Configure ASTChunk
        # Very conservative char estimate: tokens * chars_per_token * safety_buffer
        max_chunk_size_chars = int(chunk_size_tokens * 3.2 * 0.50)

        configs = {
            'max_chunk_size': max_chunk_size_chars,
            'language': language,
            'metadata_template': 'default',
            'chunk_overlap': 0,  # Disable overlap to prevent size inflation
            'chunk_expansion': False  # Avoid token-consuming metadata headers
        }

        chunk_builder = ASTChunkBuilder(**configs)

        # Use ASTChunk to chunk source code
        chunks_data = chunk_builder.chunkify(text, **configs)
        chunks = [chunk_item['content'] for chunk_item in chunks_data]

        extraction_method = f'astchunk_{language}'
        if verbose:
            print(f'  ✓ AST-aware chunking: {len(chunks)} chunks for {language} code')

        return chunks, extraction_method

    except ImportError:
        if verbose:
            print('  ⚠ ASTChunk not available, using token-aware chunking')
        chunks = chunk_text_token_aware(text, chunk_size_tokens, 0, model_name)
        return chunks, 'token_aware_fallback'

    except Exception as e:
        if verbose:
            print(f'  ⚠ ASTChunk failed ({e}), using token-aware chunking')
        chunks = chunk_text_token_aware(text, chunk_size_tokens, 0, model_name)
        return chunks, 'token_aware_fallback'

def chunk_markdown_heading_aware(
    text: str,
    chunk_size_tokens: int = 430,
    model_name: str = 'stella',
    verbose: bool = False
) -> Tuple[List[str], Optional[List[Dict]]]:
    """
    Chunk markdown text based on heading structure.
    Falls back to token-aware chunking if markdown parsing fails.

    Returns:
        Tuple of (chunks, metadata_list)
    """
    try:
        import mistune
        import re

        # Model-specific character-to-token ratios
        model_char_ratios = {
            'stella': 3.2,
            'modernbert': 3.4,
            'bge-large': 3.3,
            'default': 3.5
        }

        chars_per_token = model_char_ratios.get(model_name, 3.5)
        max_chunk_chars = int(chunk_size_tokens * chars_per_token)

        # Parse markdown using mistune's AST renderer
        md_parser = mistune.create_markdown(renderer='ast')
        ast = md_parser(text)

        def extract_text(node):
            """Recursively extract text from AST node"""
            if isinstance(node, dict):
                if node.get('type') == 'text':
                    return node.get('raw', '')
                elif 'children' in node:
                    return ''.join(extract_text(child) for child in node['children'])
            return ''

        # Simple regex-based splitting by headings
        # This is more reliable than walking the AST
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')

        chunks = []
        chunk_metadata = []
        current_chunk = []
        current_headings = []

        for line in lines:
            match = re.match(heading_pattern, line)
            if match:
                # Found a heading
                level = len(match.group(1))  # Count # symbols
                heading_text = match.group(2).strip()

                # If current chunk is large enough, save it
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > max_chunk_chars:
                    if current_chunk:
                        chunks.append(chunk_text)
                        heading_hier = ' > '.join(current_headings) if current_headings else ''
                        chunk_metadata.append({
                            'markdown_headings': heading_hier,
                            'markdown_primary_heading': current_headings[-1] if current_headings else '',
                            'markdown_section_depth': len(current_headings),
                            'markdown_heading_aware': True
                        })
                        current_chunk = []

                # Update heading hierarchy
                current_headings = current_headings[:level-1]
                current_headings.append(heading_text)

                # Add heading to chunk
                current_chunk.append(line)
            else:
                # Regular content line
                current_chunk.append(line)

                # Check if chunk is getting too large
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > max_chunk_chars * 1.2:  # Allow 20% overflow
                    if current_chunk:
                        chunks.append(chunk_text)
                        heading_hier = ' > '.join(current_headings) if current_headings else ''
                        chunk_metadata.append({
                            'markdown_headings': heading_hier,
                            'markdown_primary_heading': current_headings[-1] if current_headings else '',
                            'markdown_section_depth': len(current_headings),
                            'markdown_heading_aware': True
                        })
                        current_chunk = []

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            heading_hier = ' > '.join(current_headings) if current_headings else ''
            chunk_metadata.append({
                'markdown_headings': heading_hier,
                'markdown_primary_heading': current_headings[-1] if current_headings else '',
                'markdown_section_depth': len(current_headings),
                'markdown_heading_aware': True
            })

        if verbose:
            print(f'  ✓ Heading-aware chunking: {len(chunks)} chunks for markdown')
        return chunks, chunk_metadata

    except ImportError:
        if verbose:
            print('  ⚠ Mistune not available, using token-aware chunking')
        chunks = chunk_text_token_aware(text, chunk_size_tokens, 0, model_name)
        return chunks, None

    except Exception as e:
        if verbose:
            print(f'  ⚠ Markdown parsing failed ({e}), using token-aware chunking')
        chunks = chunk_text_token_aware(text, chunk_size_tokens, 0, model_name)
        return chunks, None

def chunk_text(
    text: str,
    file_path: str,
    store_type: str,
    chunk_size_tokens: int = 400,
    model_name: str = 'stella',
    chunk_overlap_tokens: int = 0
) -> Tuple[List[str], str, Optional[List[Dict]]]:
    """
    Main chunking function that routes to appropriate chunking strategy.

    Args:
        text: Text content to chunk
        file_path: Path to the file (for language detection)
        store_type: Type of store ('source-code', 'markdown', 'pdf', etc.)
        chunk_size_tokens: Target chunk size in tokens
        model_name: Embedding model name for token estimation

    Returns:
        Tuple of (chunks, extraction_method, optional_metadata)
    """
    # Source code: Use AST-aware chunking
    if store_type == 'source-code':
        chunks, method = chunk_source_code_ast(text, file_path, chunk_size_tokens, model_name)
        return chunks, method, None

    # Markdown: Use heading-aware chunking
    elif store_type == 'markdown' and file_path.endswith('.md'):
        chunks, metadata = chunk_markdown_heading_aware(text, chunk_size_tokens, model_name)
        return chunks, 'markdown_heading_aware' if metadata else 'token_aware', metadata

    # Everything else: Use token-aware chunking
    else:
        chunks = chunk_text_token_aware(text, chunk_size_tokens, chunk_overlap_tokens, model_name)
        return chunks, 'token_aware', None
