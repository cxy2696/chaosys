// 64-byte aligned allocator used by the ensemble storage.
//
// The velocity of the SoA kernels depends on wide, aligned loads; the standard
// allocator only guarantees alignof(max_align_t).  The allocator is stateless,
// so all instances compare equal and containers can be moved/swapped freely.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace chaosys {

template <class T, std::size_t Alignment = 64>
struct AlignedAllocator {
  static_assert(Alignment >= alignof(T), "alignment must satisfy alignof(T)");
  static_assert((Alignment & (Alignment - 1)) == 0, "alignment must be a power of two");

  using value_type = T;
  using is_always_equal = std::true_type;

  AlignedAllocator() noexcept = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    if (n > std::size_t(-1) / sizeof(T)) throw std::bad_array_new_length();

    void* raw = nullptr;
#if defined(_MSC_VER)
    raw = _aligned_malloc(n * sizeof(T), Alignment);
    if (raw == nullptr) throw std::bad_alloc();
#else
    if (::posix_memalign(&raw, Alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
#endif
    return static_cast<T*>(raw);
  }

  void deallocate(T* p, std::size_t) noexcept {
    if (p == nullptr) return;
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }

  template <class U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <class T, class U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
  return true;
}

template <class T, class U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
  return false;
}

}  // namespace chaosys
