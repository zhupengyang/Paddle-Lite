// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <mutex>  // NOLINT
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/target_wrapper.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/macros.h"

#define XPU_CALL(func)                                        \
  {                                                           \
    auto e = (func);                                          \
    CHECK_EQ(e, 0) << "XPU: (" << #func << ") returns " << e; \
  }

namespace paddle {
namespace lite {

// MAX(lod.size()) = 32
const int XPU_MAX_LOD_SIZE = 32;
// MAX(lod[i + 1] - lod[i]) = 512
const int XPU_MAX_LOD_SEQ_LEN = 512;

using TargetWrapperXPU = TargetWrapper<TARGET(kXPU)>;

struct XPUScratchPad {
  XPUScratchPad(void* addr, size_t size) : addr_(addr), size_(size) {}

  // XXX(miaotianxiang): |size_| increases monotonically
  void Reserve(size_t new_size);

  void* addr_{nullptr};
  size_t size_{0};
};

struct XPUScratchPadDeleter {
  void operator()(XPUScratchPad* sp) const;
};

using XPUScratchPadGuard = std::unique_ptr<XPUScratchPad, XPUScratchPadDeleter>;

class XpuDeviceInfo {
 public:
  static XpuDeviceInfo& Global() {
    static auto* x = new XpuDeviceInfo;
    return *x;
  }

  static std::shared_ptr<TargetWrapperXPU> GetWrapper(int device_id = 0) {
    static std::mutex mutex_conf;
    std::unique_lock<std::mutex> lck(mutex_conf);
    CHECK_GE(device_id, 0);
    if (static_cast<int>(xpu_wrappers_.size()) < device_id + 1) {
      xpu_wrappers_.resize(device_id + 1);
    }
    if (xpu_wrappers_[device_id] == nullptr) {
      xpu_wrappers_[device_id] = std::make_shared<TargetWrapperXPU>();
    }
    return xpu_wrappers_[device_id];
  }

 private:
  static std::vector<std::shared_ptr<TargetWrapperXPU>> xpu_wrappers_;
};

template <>
class TargetWrapper<TARGET(kXPU)> {
 public:
  size_t num_devices() { return 1; }
  size_t maximum_stream() { return 0; }

  void* Malloc(size_t size);
  void Free(void* ptr);

  void MemcpySync(void* dst, const void* src, size_t size, IoDirection dir);

  XPUScratchPadGuard MallocScratchPad(size_t size);

  xdnn::Context* GetRawContext() {
    if (tls_raw_ctx_ == nullptr) {
      tls_raw_ctx_ = xdnn::create_context();
      CHECK(tls_raw_ctx_);
      if (conv_autotune) {
        tls_raw_ctx_->_xpu1_conv_selector.set_autotune_loop(true);
        tls_raw_ctx_->_xpu1_conv_selector.set_inference_mode(true);
      }
      if (!conv_autotune_file.empty()) {
        tls_raw_ctx_->_xpu1_conv_selector.set_autotune_file(
            conv_autotune_file.c_str());
      }
    }
    return tls_raw_ctx_;
  }
  void MallocL3Cache();
  void FreeL3Cache();
  bool IsSharedL3Created() { return shared_l3_ptr_ == nullptr ? false : true; }
  // **DEPRECATED**, use xpu_set_device() at the very beginning of each worker
  // thread
  void SetDev(int dev_no = 0) {
    const char* dev_env = getenv("LITE_XPU_DEV");
    if (dev_env) {
      dev_no = atoi(dev_env);
    }

    XPU_CALL(xpu_set_device(dev_no));
  }

  std::string multi_encoder_precision;  // NOLINT
  size_t local_l3_size{0xfffc00};
  bool conv_autotune{false};
  std::string conv_autotune_file;  // NOLINT
  bool multi_encoder_adaptive_seqlen{false};
  size_t shared_l3_size{0};

 private:
  xdnn::Context* tls_raw_ctx_{nullptr};
  void* shared_l3_ptr_{nullptr};
  std::mutex mutex_l3_;
};

}  // namespace lite
}  // namespace paddle
