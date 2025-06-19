#pragma once
#ifdef CT2_WITH_DIRECTML
#include <mutex>

#include "dxdevice.h"

namespace ctranslate2 {
namespace dml {

class Device;

template <typename T>
class GuardedPtr {
 private:
  T* m_ptr;
  std::recursive_mutex& m_mutex;

 public:
  // Constructor: Acquires the lock
  GuardedPtr(T* ptr, std::recursive_mutex& mutex) : m_ptr(ptr), m_mutex(mutex) {
    if (!m_mutex.try_lock()) {
      throw std::invalid_argument(
          "Resource is currently locked by another thread.");
    }
    // If we get here, the lock was successful
  }

  // Destructor: Releases the lock
  ~GuardedPtr() {
    if (m_ptr) {  // Only unlock if we successfully hold a lock
      m_mutex.unlock();
    }
  }

  // --- Rule of Five: Control copying and moving ---

  // 1. No copying
  GuardedPtr(const GuardedPtr&) = delete;
  GuardedPtr& operator=(const GuardedPtr&) = delete;

  // 2. Allow moving
  GuardedPtr(GuardedPtr&& other) noexcept
      : m_ptr(other.m_ptr), m_mutex(other.m_mutex) {
    // The other object no longer owns the lock responsibility
    other.m_ptr = nullptr;
  }

  GuardedPtr& operator=(GuardedPtr&& other) = delete;

  T* operator->() const { return m_ptr; }

  T& operator*() const { return *m_ptr; }

  T* Get() const { return m_ptr; }

  // Allow checking if the pointer is valid (e.g., after a move)
  explicit operator bool() const { return m_ptr != nullptr; }
};

bool has_directml_device();
void initialize_directml();
void release_directml();  // Added

GuardedPtr<Device> get_device();

class ScopedGraphRecording {
 public:
  ScopedGraphRecording() {
    // dml::get_device()->BeginGraphRecording();
  }
  ~ScopedGraphRecording() {
    // dml::get_device()->EndGraphRecording();
  }
};

}  // namespace dml
}  // namespace ctranslate2

#endif