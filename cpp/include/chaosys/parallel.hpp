// Shared block scheduler used by both the generic ensemble path and the
// Lorenz-specific SoA kernels.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace chaosys::detail {

inline constexpr int kMaxThreads = 64;

inline int clamp_threads(int nthreads) noexcept {
  if (nthreads < 1) return 1;
  if (nthreads > kMaxThreads) return kMaxThreads;
  return nthreads;
}

// Dynamic block scheduler: each worker owns a contiguous block of trajectories
// for all time steps, so its working set stays in L1/L2.  The main thread
// participates; `n` and `block` are interpreted in trajectories.
template <typename Fn>
inline void parallel_blocks(std::size_t n, std::size_t block, int nthreads,
                            Fn fn) {
  if (n == 0) return;
  if (block == 0) block = n;
  const std::size_t nblocks = 1 + (n - 1) / block;

  int workers = clamp_threads(nthreads);
  if (static_cast<std::size_t>(workers) > nblocks) {
    workers = static_cast<int>(nblocks);
  }
  if (workers <= 1) {
    for (std::size_t b = 0; b < nblocks; ++b) {
      fn(b * block, std::min(n, (b + 1) * block));
    }
    return;
  }

  alignas(64) std::atomic<std::size_t> next{0};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  auto worker = [&]() {
    try {
      for (;;) {
        const std::size_t b = next.fetch_add(1, std::memory_order_relaxed);
        if (b >= nblocks) return;
        fn(b * block, std::min(n, (b + 1) * block));
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!failure) failure = std::current_exception();
    }
  };

  std::vector<std::thread> pool;
  try {
    pool.reserve(static_cast<std::size_t>(workers) - 1);
    for (int t = 1; t < workers; ++t) pool.emplace_back(worker);
  } catch (...) {
    // A failed thread creation must not leave joinable threads behind (their
    // destructors would call std::terminate).
    for (auto& th : pool) {
      if (th.joinable()) th.join();
    }
    throw;
  }
  worker();
  for (auto& th : pool) th.join();
  if (failure) std::rethrow_exception(failure);
}

}  // namespace chaosys::detail
