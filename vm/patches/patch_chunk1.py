"""Small first CUDA graph chunk (CHUNK1_PATCH): usage python3 patch_chunk1.py LLAMA_SRC_DIR, runtime env GGML_CUDA_GRAPH_FIRST_CHUNK=N (0 = all chunks GGML_CUDA_GRAPH_CHUNK_SIZE)."""
import os, sys

src = sys.argv[1]


def replace(rel, old, new):
    p = os.path.join(src, rel)
    s = open(p).read()
    if new in s:
        print('already replaced', rel, new[:50])
        return
    assert s.count(old) == 1, (rel, old)
    open(p, 'w').write(s.replace(old, new))
    print('replaced', rel, new[:50])


CU = 'ggml/src/ggml-cuda/ggml-cuda.cu'

replace(CU, '''        return env != nullptr ? atoi(env) : 512;
    }();
    return chunk_size;
}
''', '''        return env != nullptr ? atoi(env) : 512;
    }();
    return chunk_size;
}

// CHUNK1_PATCH: the GPU starts after the first cudaGraphLaunch returns, and its host time grows with the chunk's nodes
static int ggml_cuda_graph_first_chunk(int chunk_size) {
    static const int first = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_FIRST_CHUNK");
        return env != nullptr ? atoi(env) : 0;
    }();
    return first > 0 && first < chunk_size ? first : chunk_size;
}
''')

replace(CU, '''    int               chunk_end  = chunk_size;''',
        '''    int               chunk_end  = ggml_cuda_graph_first_chunk(chunk_size); // CHUNK1_PATCH''')
