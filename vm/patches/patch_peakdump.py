"""With GGML_ALLOC_PEAK=1, logs the largest tensors at each compute-buffer peak (diagnostic, off by default). usage: python3 patch_peakdump.py LLAMA_SRC_DIR"""
import os, sys

src = sys.argv[1]
rel = 'ggml/src/ggml-alloc.c'
p = os.path.join(src, rel)
s = open(p).read()
if 'PEAKDUMP_PATCH' in s:
    print('already patched', rel)
    sys.exit(0)

HELPERS = r'''
// PEAKDUMP_PATCH: GGML_ALLOC_PEAK=1 logs the largest live tensors at each buffer's high-water mark per reserve
struct pk_live { const void * alloc; const struct ggml_tensor * t; int chunk; size_t offset, size; };
struct pk_snap { const void * alloc; size_t max; char txt[8192]; };
static struct pk_live * pk_v;
static int pk_n, pk_cap, pk_on = -1;
static struct pk_snap pk_s[16];

static int pk_enabled(void) {
    if (pk_on < 0) {
        const char * e = getenv("GGML_ALLOC_PEAK");
        pk_on = e != NULL && atoi(e) > 0;
    }
    return pk_on;
}

static struct pk_snap * pk_slot(const void * alloc) {
    for (int i = 0; i < 16; i++) {
        if (pk_s[i].alloc == alloc) {
            return &pk_s[i];
        }
    }
    for (int i = 0; i < 16; i++) {
        if (pk_s[i].alloc == NULL) {
            pk_s[i].alloc = alloc;
            pk_s[i].max = 0;
            pk_s[i].txt[0] = 0;
            return &pk_s[i];
        }
    }
    return NULL;
}

static void pk_clear(const void * alloc) {
    for (int i = pk_n - 1; i >= 0; i--) {
        if (pk_v[i].alloc == alloc) {
            pk_v[i] = pk_v[--pk_n];
        }
    }
    struct pk_snap * sn = pk_slot(alloc);
    if (sn) {
        sn->max = 0;
        sn->txt[0] = 0;
    }
}

static void pk_add(const void * alloc, const struct ggml_tensor * t, int chunk, size_t offset, size_t size) {
    if (pk_n == pk_cap) {
        pk_cap = pk_cap ? 2*pk_cap : 4096;
        pk_v = (struct pk_live *) realloc(pk_v, pk_cap*sizeof(*pk_v));
        GGML_ASSERT(pk_v != NULL);
    }
    pk_v[pk_n].alloc = alloc;
    pk_v[pk_n].t = t;
    pk_v[pk_n].chunk = chunk;
    pk_v[pk_n].offset = offset;
    pk_v[pk_n].size = size;
    pk_n++;
}

static void pk_del(const void * alloc, int chunk, size_t offset) {
    for (int i = pk_n - 1; i >= 0; i--) {
        if (pk_v[i].alloc == alloc && pk_v[i].chunk == chunk && pk_v[i].offset == offset) {
            pk_v[i] = pk_v[--pk_n];
            return;
        }
    }
}

static void pk_snapshot(const void * alloc, size_t max) {
    struct pk_snap * sn = pk_slot(alloc);
    if (!sn) {
        return;
    }
    int top[24];
    int nt = 0, n = 0;
    size_t sum = 0;
    for (int i = 0; i < pk_n; i++) {
        if (pk_v[i].alloc != alloc) {
            continue;
        }
        n++;
        sum += pk_v[i].size;
        int j;
        if (nt < 24) {
            j = nt++;
        } else if (pk_v[i].size > pk_v[top[23]].size) {
            j = 23;
        } else {
            continue;
        }
        while (j > 0 && pk_v[top[j - 1]].size < pk_v[i].size) {
            top[j] = top[j - 1];
            j--;
        }
        top[j] = i;
    }
    sn->max = max;
    int o = snprintf(sn->txt, sizeof(sn->txt), "max=%.1f MiB live=%d sum=%.1f MiB:", max/1048576.0, n, sum/1048576.0);
    for (int k = 0; k < nt && o > 0 && o < (int) sizeof(sn->txt) - 1; k++) {
        const struct ggml_tensor * t = pk_v[top[k]].t;
        o += snprintf(sn->txt + o, sizeof(sn->txt) - o, " %s|%s|%lldx%lldx%lldx%lld|%s=%.1f", t->name, ggml_op_desc(t),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type),
                pk_v[top[k]].size/1048576.0);
    }
}

'''


def sub(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


sub('static void ggml_dyn_tallocr_insert_block(', HELPERS + 'static void ggml_dyn_tallocr_insert_block(')
sub('    chunk->max_size = MAX(chunk->max_size, addr.offset + size);\n',
    '''    if (pk_enabled()) {
        pk_add(alloc, tensor, addr.chunk, addr.offset, size);
        if (addr.offset + size > chunk->max_size) {
            size_t tot = addr.offset + size;
            for (int c = 0; c < alloc->n_chunks; c++) {
                tot += alloc->chunks[c] != chunk ? alloc->chunks[c]->max_size : 0;
            }
            pk_snapshot(alloc, tot);
        }
    }
    chunk->max_size = MAX(chunk->max_size, addr.offset + size);
''')
sub('static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {\n',
    'static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {\n'
    '    if (pk_enabled()) {\n        pk_del(alloc, addr.chunk, addr.offset);\n    }\n')
sub('static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {\n',
    'static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {\n    if (pk_enabled()) {\n        pk_clear(alloc);\n    }\n')
sub('    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);\n',
    '''    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    if (pk_enabled()) {
        for (int i = 0; i < galloc->n_buffers; i++) {
            struct pk_snap * sn = pk_slot(galloc->buf_tallocs[i]);
            if (sn && sn->max > 0) {
                fprintf(stderr, "PEAK %s nodes=%d %s\\n", ggml_backend_buft_name(galloc->bufts[i]), graph->n_nodes, sn->txt);
            }
        }
    }
''')
open(p, 'w').write(s)
for mark in ['PEAKDUMP_PATCH', 'pk_add(alloc, tensor', 'pk_del(alloc, addr.chunk', 'pk_clear(alloc)', 'fprintf(stderr, "PEAK']:
    assert mark in open(p).read(), mark
print('PEAKDUMP patch complete')
