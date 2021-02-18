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

#include "lite/kernels/xpu/concat_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void ConcatCompute<T>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto ins = param.x;
  auto out = param.output;
  int64_t axis = param.axis;

  std::vector<const T*> x_list;
  std::vector<std::vector<int>> xdims_list;
  for (size_t i = 0; i < ins.size(); i++) {
    xdims_list.push_back(std::vector<int>());
    for (size_t j = 0; j < ins[i]->dims().size(); j++) {
      xdims_list[i].push_back(ins[i]->dims()[j]);
    }
    x_list.push_back(ins[i]->template data<T>());
  }

  int r = xdnn::concat<T>(ctx.GetRawContext(),
                          x_list,
                          out->template mutable_data<T>(TARGET(kXPU)),
                          xdims_list,
                          axis);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using ConcatFloat32 = paddle::lite::kernels::xpu::ConcatCompute<float>;
REGISTER_LITE_KERNEL(concat, kXPU, kAny, kAny, ConcatFloat32, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using ConcatInt64 = paddle::lite::kernels::xpu::ConcatCompute<int64_t>;
REGISTER_LITE_KERNEL(concat, kXPU, kAny, kAny, ConcatInt64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
