// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/increment_compute.h"
#include <vector>
#include "lite/backends/xpu/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void increment(const T* input, const int n, const T step, T* out) {
  for (int i = 0; i < n; i++) {
    out[i] = input[i] + step;
  }
}

#define _Increment(type, dtype)                                              \
  case PRECISION(type): {                                                    \
    const auto* x_data = param.X->data<dtype>();                             \
    std::vector<dtype> x_vec(total_num);                                     \
    TargetWrapperXPU::MemcpySync(                                            \
        &(x_vec[0]), x_data, total_num * sizeof(dtype), IoDirection::DtoH);  \
    dtype step = static_cast<dtype>(param.step);                             \
    increment(x_vec.data(), total_num, step, &(x_vec[0]));                   \
    auto* o_data = param.Out->mutable_data<dtype>();                         \
    TargetWrapperXPU::MemcpySync(                                            \
        o_data, x_vec.data(), total_num * sizeof(dtype), IoDirection::HtoD); \
    break;                                                                   \
  }

void IncrementCompute::Run() {
  auto& param = this->Param<operators::IncrementParam>();

  int total_num = param.X->numel();
  switch (param.X->precision()) {
    _Increment(kInt64, int64_t) _Increment(kInt32, int)
            _Increment(kFloat, float) default
        : LOG(FATAL)
          << "unsupport input type "
          << PrecisionToStr(param.X->precision());
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(increment,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::IncrementCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kAny))})
    .Finalize();
