// LD_PRELOAD shim: records mmaps of model files and, on request, pins chosen ranges with cudaHostRegister.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#define MAXMAP 64
static struct { void *base; size_t len; char path[PATH_MAX]; } maps[MAXMAP];
static int nmaps;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static const char *req_path = "/content/res/pin_request";
static const char *res_path = "/content/res/pin_result";

static void *watcher(void *arg) {
    (void)arg;
    for (;;) {
        FILE *f = fopen(req_path, "r");
        if (!f) { sleep(1); continue; }
        void *h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libcudart.so.13", RTLD_NOW | RTLD_NOLOAD);
        int (*reg)(void *, size_t, unsigned) = h ? (int (*)(void *, size_t, unsigned))dlsym(h, "cudaHostRegister") : NULL;
        const char *pm = getenv("PIN_MAPPED");
        const unsigned flags = 0x01 | 0x08 | (pm && atoi(pm) ? 0x02 : 0);  // Portable | ReadOnly, PIN_MAPPED=1 adds Mapped (patch_copybatch)
        FILE *out = fopen(res_path, "w");
        char path[PATH_MAX];
        unsigned long long off, len;
        size_t total = 0;
        int fails = 0, nomap = 0;
        while (fscanf(f, "%4095s %llu %llu", path, &off, &len) == 3) {
            int rc = -1;
            pthread_mutex_lock(&mu);
            for (int i = 0; i < nmaps; i++)
                if (!strcmp(maps[i].path, path) && off + len <= maps[i].len && reg) {
                    rc = reg((char *)maps[i].base + off, len, flags);
                    if (rc != 0 && flags != (0x01 | 0x08)) {
                        rc = reg((char *)maps[i].base + off, len, 0x01 | 0x08);
                        nomap += rc == 0;
                    }
                }
            pthread_mutex_unlock(&mu);
            if (rc == 0) total += len; else fails++;
            if (out) fprintf(out, "%s %llu %llu rc=%d\n", path, off, len, rc);
        }
        fclose(f);
        unlink(req_path);
        if (out) { fprintf(out, "DONE pinned=%zu fails=%d cudart=%s flags=%u unmapped=%d\n", total, fails, reg ? "ok" : "missing", flags, nomap); fclose(out); }
    }
    return NULL;
}

static void start_watcher(void) {
    pthread_t t;
    pthread_create(&t, NULL, watcher, NULL);
    pthread_detach(t);
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    void *p = real_mmap(addr, len, prot, flags, fd, off);
    if (p != MAP_FAILED && fd >= 0 && off == 0 && len > (1UL << 30)) {
        char link[64], path[PATH_MAX];
        snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(link, path, sizeof path - 1);
        if (n > 0) {
            path[n] = 0;
            pthread_mutex_lock(&mu);
            if (nmaps < MAXMAP) { maps[nmaps].base = p; maps[nmaps].len = len; strcpy(maps[nmaps].path, path); nmaps++; }
            pthread_mutex_unlock(&mu);
            pthread_once(&once, start_watcher);
        }
    }
    return p;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) { return mmap(addr, len, prot, flags, fd, off); }
