"""Graph inputs uploaded without host syncs (ASYNCIN_PATCH): usage python3 patch_asyncin.py LLAMA_SRC_DIR, runtime env GGML_SCHED_ASYNC_INPUTS=1."""
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


BE = 'ggml/src/ggml-backend.cpp'

replace(BE, '''    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;''', '''    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    // ASYNCIN_PATCH: pinned staging for user inputs, one half per compute, a half is reused after the event of the compute two back
    ggml_backend_buffer_t in_stage[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_event_t  in_stage_ev[GGML_SCHED_MAX_BACKENDS][2];
    size_t                in_stage_off[GGML_SCHED_MAX_BACKENDS];
    int                   in_stage_half[GGML_SCHED_MAX_BACKENDS];
    bool                  in_stage_open[GGML_SCHED_MAX_BACKENDS];
    bool                  in_stage_failed[GGML_SCHED_MAX_BACKENDS];
    struct ggml_tensor ** graph_inputs;''')

replace(BE, '''static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {''',
        '''#define GGML_SCHED_IN_STAGE_HALF ((size_t) 32*1024*1024)

static bool ggml_backend_sched_async_inputs(void) {
    static const bool on = [] {
        const char * env = getenv("GGML_SCHED_ASYNC_INPUTS");
        return env != nullptr && atoi(env) != 0;
    }();
    return on;
}

// a pinned slot of the current half for a user input, NULL when the input does not fit or the backend has no pinned host memory or events
static void * ggml_backend_sched_stage(ggml_backend_sched_t sched, int id, size_t size) {
    if (sched->in_stage_failed[id]) {
        return NULL;
    }
    if (sched->in_stage[id] == NULL) {
        ggml_backend_dev_t         dev = ggml_backend_get_device(sched->backends[id]);
        ggml_backend_buffer_type_t hbt = dev ? ggml_backend_dev_host_buffer_type(dev) : NULL;
        for (int h = 0; h < 2; h++) {
            sched->in_stage_ev[id][h] = hbt ? ggml_backend_event_new(dev) : NULL;
        }
        if (sched->in_stage_ev[id][0] != NULL && sched->in_stage_ev[id][1] != NULL) {
            sched->in_stage[id] = ggml_backend_buft_alloc_buffer(hbt, 2*GGML_SCHED_IN_STAGE_HALF);
        }
        if (sched->in_stage[id] == NULL) {
            for (int h = 0; h < 2; h++) {
                ggml_backend_event_free(sched->in_stage_ev[id][h]);
                sched->in_stage_ev[id][h] = NULL;
            }
            sched->in_stage_failed[id] = true;
            GGML_LOG_WARN("%s: no pinned staging for %s, graph inputs are copied synchronously\\n", __func__, ggml_backend_name(sched->backends[id]));
            return NULL;
        }
        GGML_LOG_INFO("%s: graph inputs for %s are staged in 2 x %zu MiB of pinned memory and uploaded asynchronously\\n", __func__,
            ggml_backend_name(sched->backends[id]), GGML_SCHED_IN_STAGE_HALF/(1024*1024));
    }
    if (!sched->in_stage_open[id]) {
        sched->in_stage_half[id] ^= 1;
        ggml_backend_event_synchronize(sched->in_stage_ev[id][sched->in_stage_half[id]]);
        sched->in_stage_off[id]  = 0;
        sched->in_stage_open[id] = true;
    }
    if (sched->in_stage_off[id] + size > GGML_SCHED_IN_STAGE_HALF) {
        return NULL;
    }
    char * p = (char *) ggml_backend_buffer_get_base(sched->in_stage[id]) + sched->in_stage_half[id]*GGML_SCHED_IN_STAGE_HALF + sched->in_stage_off[id];
    sched->in_stage_off[id] += GGML_PAD(size, 256);
    return p;
}

static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {''')

replace(BE, '''    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {''', '''    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // ASYNCIN_PATCH: staged now, so the user may overwrite the data; the upload is ordered on the split backend's stream before its graph
        void * stage = NULL;
        if (ggml_backend_sched_async_inputs() && sched->n_copies == 1 && split_backend->iface.set_tensor_async != NULL &&
                input->buffer != NULL && ggml_backend_buffer_is_host(input->buffer) &&
                (stage = ggml_backend_sched_stage(sched, split_backend_id, ggml_nbytes(input))) != NULL) {
            memcpy(stage, input->data, ggml_nbytes(input));
            ggml_backend_tensor_set_async(split_backend, input_cpy, stage, 0, ggml_nbytes(input));
            return;
        }
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {''')

replace(BE, '''        prev_backend_id = split_backend_id;
    }

    return GGML_STATUS_SUCCESS;
}''', '''        prev_backend_id = split_backend_id;
    }

    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->in_stage_open[b]) { // ASYNCIN_PATCH
            ggml_backend_event_record(sched->in_stage_ev[b][sched->in_stage_half[b]], sched->backends[b]);
            sched->in_stage_open[b] = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}''')

replace(BE, '''    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }''', '''    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        if (sched->in_stage[b] != NULL) { // ASYNCIN_PATCH
            ggml_backend_synchronize(sched->backends[b]);
            ggml_backend_buffer_free(sched->in_stage[b]);
            ggml_backend_event_free(sched->in_stage_ev[b][0]);
            ggml_backend_event_free(sched->in_stage_ev[b][1]);
        }
    }''')
