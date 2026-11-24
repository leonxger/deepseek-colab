#!/usr/bin/env python3
"""Surgical Colab G4 patches for jesdga95/basalt v0.5.7 (BASALT_COLAB_PATCH).

Applies three opt-in/fallback patches so Basalt DeepSeek-V4.1-Flash packs boot in ~2 min and
serve at full speed on Colab G4 (96 GiB VRAM + 176 GiB RAM + 232 GB Disk):
  1. src/models/deepseek41/model.cpp (BASALT_COLAB_ENGRAM_ENV):
     When BASALT_ENGRAM_FILE is set, open the 110.6 GB Engram table from that separate path at
     offset 0 (e.g. /mnt/ram/engram.bin on tmpfs) instead of inside the sparse .basalt backbone.
  2. src/platform/direct_file.cpp (BASALT_COLAB_TMPFS_FALLBACK):
     When open(..., O_RDONLY | O_DIRECT | O_CLOEXEC) fails with EINVAL (tmpfs / overlayfs),
     fall back cleanly to O_RDONLY | O_CLOEXEC.
  3. src/core/expert_tier.cpp & src/core/loader.cpp (BASALT_COLAB_PUNCH_ON_READ):
     When BASALT_PUNCH_FILE is set (e.g. /mnt/ram/model.basalt on tmpfs):
       - expert_tier.cpp credits the file's punchable tmpfs blocks (st_blocks*512 minus 2 GiB
         kept for header/meta/token_embd/vision) to `avail` so the RAM expert tier plans the
         full ~54 GiB complement before weights are punched.
       - loader.cpp executes device (VRAM) reads first, punches all inward-4K-aligned spans of
         (device_reads \\ host_reads) via fallocate(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE)
         (~66 GiB freed immediately), then executes host (RAM) reads and punches host_reads
         (~54 GiB freed immediately) — dropping /mnt/ram/model.basalt from 140 GiB -> ~2 GiB
         before pin_landed_complement() and the 110.6 GB Engram download.

Usage: python3 patch_basalt.py BASALT_SRC_DIR
"""
import os, sys


def patch_file(src_dir, rel, marker, edits):
    path = os.path.join(src_dir, rel)
    text = open(path, 'r', encoding='utf-8').read()
    if marker in text:
        print(f'already patched {rel} ({marker})')
        return
    for old, new in edits:
        count = text.count(old)
        assert count == 1, f'{rel}: expected 1 match (got {count}) for:\n{old[:220]}'
        text = text.replace(old, new, 1)
    open(path, 'w', encoding='utf-8').write(text)
    print(f'patched {rel} ({marker})')


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python3 patch_basalt.py BASALT_SRC_DIR')
    src = sys.argv[1]

    # 1. Separate Engram file via BASALT_ENGRAM_FILE
    patch_file(
        src,
        'src/models/deepseek41/model.cpp',
        'BASALT_COLAB_ENGRAM_ENV',
        [
            (
                '''    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        if (!engram_->open(path, e.file_offset, e.rows, 64, 1 << 16, err, e.row_bytes)) return false;
    }''',
                '''    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        // BASALT_COLAB_ENGRAM_ENV: allow serving the 110.6 GB Engram table from a separate tmpfs file
        const char* engram_env = std::getenv("BASALT_ENGRAM_FILE");
        const std::string engram_path = (engram_env && *engram_env) ? std::string(engram_env) : path;
        const uint64_t engram_off = (engram_env && *engram_env) ? 0ull : e.file_offset;
        if (!engram_->open(engram_path, engram_off, e.rows, 64, 1 << 16, err, e.row_bytes)) return false;
    }''',
            ),
        ],
    )

    # 2. DirectFile EINVAL fallback for tmpfs / overlayfs
    patch_file(
        src,
        'src/platform/direct_file.cpp',
        'BASALT_COLAB_TMPFS_FALLBACK',
        [
            (
                '#include <cstdlib>',
                '#include <cerrno>\n#include <cstdlib>',
            ),
            (
                '''    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }''',
                '''    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (impl_->fd < 0 && errno == EINVAL) { // BASALT_COLAB_TMPFS_FALLBACK: tmpfs rejects O_DIRECT
        impl_->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    }
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }''',
            ),
        ],
    )

    # 3a. ExpertTier: credit punchable tmpfs backbone bytes into avail when BASALT_PUNCH_FILE is active
    patch_file(
        src,
        'src/core/expert_tier.cpp',
        'BASALT_COLAB_PUNCH_ON_READ',
        [
            (
                '#include <cstring>',
                '#include <cstring>\n#ifdef __linux__\n#include <sys/stat.h>\n#endif',
            ),
            (
                '''    uint64_t avail = available();
    // places per layer above the reserve, less the spare each layer keeps, in `bytes` of RAM''',
                '''    uint64_t avail = available();
    // BASALT_COLAB_PUNCH_ON_READ: credit tmpfs backbone blocks that Loader::run will punch before RAM reads
#ifdef __linux__
    uint64_t punch_credit = 0;
    if (const char* pf = std::getenv("BASALT_PUNCH_FILE"); pf && *pf) {
        struct stat st{};
        if (::stat(pf, &st) == 0 && st.st_blocks > 0) {
            const uint64_t phys = (uint64_t) st.st_blocks * 512ull;
            constexpr uint64_t kKeepBytes = 2ull << 30;
            if (phys > kKeepBytes) punch_credit = phys - kKeepBytes;
        }
    }
    const uint64_t phys_cap = std::min(platform::total_physical_memory(), hm.cgroup_limit);
    if (punch_credit > 0) {
        avail = std::min(avail + punch_credit, phys_cap > host_bytes_ ? phys_cap - host_bytes_ : avail);
    }
#endif
    // places per layer above the reserve, less the spare each layer keeps, in `bytes` of RAM''',
            ),
        ],
    )

    # 3b. Loader: two-wave barriered execution + inward-4K fallocate(PUNCH_HOLE) on BASALT_PUNCH_FILE
    patch_file(
        src,
        'src/core/loader.cpp',
        'BASALT_COLAB_PUNCH_ON_READ',
        [
            (
                '#include <cstring>',
                '#include <cstdlib>\n#include <cstring>\n#ifdef __linux__\n#include <fcntl.h>\n#include <sys/stat.h>\n#include <unistd.h>\n#endif',
            ),
            (
                '''    const size_t device_tasks = rounds[0].size();
    rounds[0].insert(rounds[0].end(), host_reads.begin(), host_reads.end());''',
                '''    // BASALT_COLAB_PUNCH_ON_READ: when punching a tmpfs model file on read, finish all device reads first,
    // punch device-only byte ranges to reclaim ~66 GiB RAM, then run host reads and punch host ranges (~54 GiB).
    int punch_fd = -1;
    std::vector<bool> file_is_punch(handles.size(), false);
#ifdef __linux__
    if (const char* pf = std::getenv("BASALT_PUNCH_FILE"); pf && *pf) {
        struct stat target_st{};
        if (::stat(pf, &target_st) == 0) {
            bool any_match = false;
            for (size_t i = 0; i < handles.size(); ++i) {
                struct stat f_st{};
                const int fd_chk = handles[i]->buffered >= 0 ? handles[i]->buffered : handles[i]->direct;
                if (fd_chk >= 0 && ::fstat(fd_chk, &f_st) == 0 &&
                    f_st.st_dev == target_st.st_dev && f_st.st_ino == target_st.st_ino) {
                    file_is_punch[i] = true;
                    any_match = true;
                }
            }
            if (any_match) {
                punch_fd = ::open(pf, O_RDWR | O_CLOEXEC);
                if (punch_fd >= 0) early_ram = 0;
            }
        }
    }
#endif
    const size_t device_tasks = rounds[0].size();
    std::vector<Task> punch_host_wave;
    if (punch_fd >= 0 && device_tasks > 0 && !host_reads.empty()) {
        punch_host_wave = std::move(host_reads);
    } else {
        rounds[0].insert(rounds[0].end(), host_reads.begin(), host_reads.end());
    }''',
            ),
            (
                '        if (device_tasks < rounds[0].size()) stages.push_back("ram");',
                '        if (device_tasks < rounds[0].size() || !punch_host_wave.empty()) stages.push_back("ram");',
            ),
            (
                '''    for (const std::vector<Task>& tasks : rounds) {   // each round's copies are complete before the next round reads them
        if (&tasks == &rounds[1] && !failed.load())   // the reads have landed: the plan's steps, before the copies
            for (const auto& step : plan.steps()) {
                std::string e;
                if (!step(e)) { fail(e); break; }
            }
        if (tasks.empty() || failed.load()) continue;
        std::atomic<size_t> next{0};
        const int n = (int) std::min<size_t>((size_t) n_workers, tasks.size());
        std::vector<std::thread> pool;
        for (int i = 1; i < n; ++i) pool.emplace_back(work, i, std::cref(tasks), std::ref(next));
        work(0, tasks, next);
        for (auto& t : pool) t.join();
    }
    if (any_dev) (void) cudaSetDevice(prev_device);''',
                '''    auto run_wave = [&](const std::vector<Task>& wave) {
        if (wave.empty() || failed.load()) return;
        std::atomic<size_t> next{0};
        const int n = (int) std::min<size_t>((size_t) n_workers, wave.size());
        std::vector<std::thread> pool;
        for (int i = 1; i < n; ++i) pool.emplace_back(work, i, std::cref(wave), std::ref(next));
        work(0, wave, next);
        for (auto& t : pool) t.join();
    };
#ifdef __linux__
    using Span = std::pair<uint64_t, uint64_t>;
    auto merge_spans = [](std::vector<Span>& sp) {
        if (sp.empty()) return;
        std::sort(sp.begin(), sp.end());
        size_t w = 0;
        for (size_t i = 1; i < sp.size(); ++i) {
            if (sp[i].first <= sp[w].second) sp[w].second = std::max(sp[w].second, sp[i].second);
            else sp[++w] = sp[i];
        }
        sp.resize(w + 1);
    };
    auto collect_spans = [&](const std::vector<Task>& wave) {
        std::vector<Span> sp;
        for (const Task& t : wave) {
            const LoadItem& it = items[t.item];
            if (it.from == nullptr && it.file >= 0 && file_is_punch[(size_t) it.file] && t.len > 0)
                sp.emplace_back(it.off + t.off, it.off + t.off + t.len);
        }
        merge_spans(sp);
        return sp;
    };
    auto punch_spans = [&](const std::vector<Span>& sp, const std::vector<Span>& keep, const char* tag) {
        if (punch_fd < 0 || sp.empty()) return;
        uint64_t punched = 0;
        size_t ki = 0;
        auto do_punch = [&](uint64_t a, uint64_t b) {
            const uint64_t pa = (a + 4095ull) & ~4095ull;   // inward 4K alignment: never touches adjacent tensors
            const uint64_t pb = b & ~4095ull;
            if (pb > pa && ::fallocate(punch_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                                       (off_t) pa, (off_t) (pb - pa)) == 0)
                punched += (pb - pa);
        };
        for (auto [a, b] : sp) {
            while (ki < keep.size() && keep[ki].second <= a) ++ki;
            size_t j = ki;
            uint64_t cur = a;
            while (j < keep.size() && keep[j].first < b) {
                if (keep[j].first > cur) do_punch(cur, keep[j].first);
                cur = std::max(cur, keep[j].second);
                if (cur >= b) break;
                ++j;
            }
            if (cur < b) do_punch(cur, b);
        }
        if (punched > 0)
            basalt::log::info("basalt loader", "punch-on-read (%s): freed %.2f GiB tmpfs",
                              tag, (double) punched / 1073741824.0);
    };
#endif

    run_wave(rounds[0]);
#ifdef __linux__
    if (punch_fd >= 0 && !failed.load()) {
        if (!punch_host_wave.empty()) {
            const auto dev_sp  = collect_spans(rounds[0]);
            const auto host_sp = collect_spans(punch_host_wave);
            punch_spans(dev_sp, host_sp, "after-vram");
            LoadProgress::into("ram");
            basalt::log::info("basalt loader", "the cards' share is read (%.2f GiB in %.1f s); RAM next",
                              (double) device_bytes / 1073741824.0,
                              std::chrono::duration<double>(std::chrono::steady_clock::now() - run_t0).count());
            run_wave(punch_host_wave);
            if (!failed.load()) punch_spans(host_sp, {}, "after-ram");
        } else {
            const auto all_sp = collect_spans(rounds[0]);
            punch_spans(all_sp, {}, "after-reads");
        }
    }
    if (punch_fd >= 0) ::close(punch_fd);
#endif

    if (!failed.load())   // the reads have landed: the plan's steps, before the copies
        for (const auto& step : plan.steps()) {
            std::string e;
            if (!step(e)) { fail(e); break; }
        }
    run_wave(rounds[1]);
    if (any_dev) (void) cudaSetDevice(prev_device);''',
            ),
        ],
    )

    # 4. Serve guard: refuse unload and refuse engine respawn after BASALT_PUNCH_FILE punched the tmpfs pack
    patch_file(
        src,
        'serve/service.py',
        'BASALT_COLAB_PUNCH_GUARD',
        [
            (
                'import subprocess\n',
                'import os\nimport subprocess\n',
            ),
            (
                '''        try:
            self._ensure_loaded()
            if self.on_loaded is not None:
                self.on_loaded()''',
                '''        try:
            self._ensure_loaded()
            if os.environ.get("BASALT_PUNCH_FILE"):
                self._punched_once = True
            if self.on_loaded is not None:
                self.on_loaded()''',
            ),
            (
                '''        if self.loaded() and not self._vision_down():
            return''',
                '''        if self.loaded() and not self._vision_down():
            return
        # BASALT_COLAB_PUNCH_GUARD: punched tmpfs weights cannot be re-read after an engine exit
        if os.environ.get("BASALT_PUNCH_FILE") and getattr(self, "_punched_once", False):
            raise EngineDied("model file was punched on read; restart the VM to reload")''',
            ),
            (
                '''        if not hasattr(self.engine, "unload"):
            return "unsupported"''',
                '''        if not hasattr(self.engine, "unload") or os.environ.get("BASALT_PUNCH_FILE"):
            return "unsupported"''',
            ),
        ],
    )


if __name__ == '__main__':
    main()
