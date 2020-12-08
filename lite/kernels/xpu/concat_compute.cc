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
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ConcatCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto ins = param.x;
  auto out = param.output;
  int64_t axis = param.axis;

  std::vector<const float*> x_list;
  std::vector<std::vector<int>> xdims_list;
  for (int i = 0; i < ins.size(); i++) {
    xdims_list.push_back(std::vector<int>());
    for (int j = 0; j < ins[i]->dims().size(); j++) {
      xdims_list[i].push_back(ins[i]->dims()[j]);
    }
    x_list.push_back(ins[i]->data<float>());
  }

  for (size_t i = 0; i < x_list.size(); i++) {
    Tensor tmp;
    tmp.Resize(ins[i]->dims());
    auto tmp_data = tmp.mutable_data<float>();
    TargetWrapperXPU::MemcpySync(tmp_data,
                                 ins[i]->raw_data(),
                                 sizeof(float) * ins[i]->numel(),
                                 IoDirection::DtoH);
    float sum = 0.;
    for (int64_t j = 0; j < ins[i]->numel(); j++) {
      sum += tmp_data[j];
    }
    LOG(INFO) << "--- concat in " << i << " : " << tmp_data[0] << ", "
              << tmp_data[1] << ", " << tmp_data[2] << ", sum: " << sum;
  }

  int r = xdnn::concat<float>(ctx.GetRawContext(),
                              x_list,
                              out->mutable_data<float>(TARGET(kXPU)),
                              xdims_list,
                              axis);

  CHECK_EQ(r, 0);

  Tensor tmp;
  tmp.Resize(param.output->dims());
  auto tmp_data = tmp.mutable_data<float>();
  TargetWrapperXPU::MemcpySync(tmp_data,
                               param.output->raw_data(),
                               sizeof(float) * param.output->numel(),
                               IoDirection::DtoH);
  LOG(INFO) << "--- concat out: " << tmp_data[0] << ", " << tmp_data[1] << ", "
            << tmp_data[2] << ", ";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    concat, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::ConcatCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
