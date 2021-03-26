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

#include "lite/kernels/xpu/assign_compute.h"
#include <algorithm>
#include "lite/backends/xpu/xpu_header_sitter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void AssignCompute::Run() {
  auto& param = Param<param_t>();
  CHECK(param.X) << "only support input is tensor";
  if (param.X == param.Out) {
    return;
  }

  auto& ctx = this->ctx_->As<XPUContext>();
  int r;
  switch (param.X->precision()) {
    case PRECISION(kFloat): {
      r = xdnn::copy<float>(ctx.GetRawContext(),
                            param.X->data<float>(),
                            param.Out->mutable_data<float>(TARGET(kXPU)),
                            param.X->numel());
      break;
    }
    case PRECISION(kInt32): {
      r = xdnn::copy<int>(ctx.GetRawContext(),
                          param.X->data<int>(),
                          param.Out->mutable_data<int>(TARGET(kXPU)),
                          param.X->numel());
      break;
    }
    case PRECISION(kInt64): {
      r = xdnn::copy<int64_t>(ctx.GetRawContext(),
                              param.X->data<int64_t>(),
                              param.Out->mutable_data<int64_t>(TARGET(kXPU)),
                              param.X->numel());
      break;
    }
    default:
      LOG(FATAL) << "unsupported date type: "
                 << lite_api::PrecisionToStr(param.X->precision());
  }

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    assign, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::AssignCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
