// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/mir/weight_to_device_pass.h"
#include <utility>
#include <vector>
#include "lite/core/mir/pass_registry.h"
// #include "lite/core/program.h"

namespace paddle {
namespace lite {
namespace mir {

void WeightToDevicePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // std::list<Node*> nodes;
  std::vector<Instruction> insts;
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->IsStmt() && node->AsStmt().op_type() == "io_copy_once") {
      auto stmt = node->AsStmt();
      stmt.op()->InferShape();
      stmt.kernels().front()->Launch();
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(weight_to_device_pass, paddle::lite::mir::WeightToDevicePass)
    .BindTargets({TARGET(kXPU)})
    .ExcludeTargets({TARGET(kARM),
                     TARGET(kOpenCL),
                     TARGET(kNPU),
                     TARGET(kBM),
                     TARGET(kRKNPU),
                     TARGET(kAPU),
                     TARGET(kMLU),
                     TARGET(kHuaweiAscendNPU),
                     TARGET(kImaginationNNA)});
