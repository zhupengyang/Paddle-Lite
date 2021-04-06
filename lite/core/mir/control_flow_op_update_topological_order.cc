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

#include <map>
#include <unordered_set>
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class ControlFlowOpUpdateTopologicalOrderPass : public mir::StmtPass {
 public:
  void SetAllGraphs(std::vector<std::unique_ptr<mir::SSAGraph>> *graphs) {
    CHECK(graphs && !graphs->empty());
    graphs_ = graphs;
  }

  void Apply(const std::unique_ptr<SSAGraph> &graph) override {
    const std::unordered_set<std::string> control_flow_op_types = {
        "while", "conditional_block"};
    std::map<int, Node *> control_flow_ops;
    for (auto &op_node : graph->mutable_nodes()) {
      if (op_node.IsArg()) continue;
    }
  }

 private:
  std::vector<std::unique_ptr<mir::SSAGraph>> *graphs_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(control_flow_op_update_topological_order,
                  paddle::lite::mir::ControlFlowOpUpdateTopologicalOrderPass)
    .BindTargets({TARGET(kXPU)});
