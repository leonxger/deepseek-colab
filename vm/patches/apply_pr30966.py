"""Applies llama.cpp PR #30966 (MoE expert cache, ggml-org aman/moe-cache-cpu-misses @ ae36884b) to a b11963 tree, idempotent. usage: python3 apply_pr30966.py LLAMA_SRC_DIR"""
import os, subprocess, sys

src, diff = sys.argv[1], os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pr30966.diff')
ex = ['--exclude=CODEOWNERS', '--exclude=docs/*', '--exclude=scripts/*', '--exclude=tools/*/README.md']
if subprocess.run(['git', 'apply', '--reverse', '--check'] + ex + [diff], cwd=src, capture_output=True).returncode == 0:
    print('already applied pr30966')
    sys.exit(0)
subprocess.run(['git', 'apply'] + ex + [diff], cwd=src, check=True)
print('applied pr30966')
