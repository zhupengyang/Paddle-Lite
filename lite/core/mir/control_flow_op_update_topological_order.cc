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

#include "lite/core/mir/control_flow_op_update_topological_order.h"
#include <algorithm>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

namespace paddle {
namespace lite {
namespace mir {

void ControlFlowOpUpdateTopologicalOrderPass::SetAllGraphs(
    std::vector<std::unique_ptr<mir::SSAGraph>> *graphs) {
  CHECK(graphs && !graphs->empty());
  graphs_ = graphs;
}

void ControlFlowOpUpdateTopologicalOrderPass::Apply(
    const std::unique_ptr<SSAGraph> &graph) {
  const std::unordered_set<std::string> control_flow_op_types = {
      "while", "conditional_block"};
  std::vector<std::pair<int, Node *>> control_flow_ops;
  for (auto &op_node : graph->mutable_nodes()) {
    if (op_node.IsArg()) continue;
    auto *stmt = op_node.stmt();
    const std::string op_type = stmt->op_type();
    if (control_flow_op_types.count(op_type) == 0) continue;
    int block_id = stmt->op_info()->GetAttr<int>("sub_block");
    control_flow_ops.push_back(std::make_pair(block_id, &op_node));
  }

  auto cmp_cotronl_flow_op = [](const std::pair<int, Node *> &a,
                                const std::pair<int, Node *> &b) {
    return a.first < b.first;
  };
  std::sort(
      control_flow_ops.begin(), control_flow_ops.end(), cmp_cotronl_flow_op);
  for (int i = 0; i < static_cast<int>(control_flow_ops.size()) - 1; i++) {
    auto *new_var_node = graph->NewArgumentNode(
        "control_flow_op_" + std::to_string(control_flow_ops[i].first) +
        "_to_" + std::to_string(control_flow_ops[i + 1].first));
    auto *op_node_1 = control_flow_ops[i].second;
    auto *op_node_2 = control_flow_ops[i + 1].second;
    IR_NODE_LINK_TO(op_node_1, new_var_node);
    IR_NODE_LINK_TO(new_var_node, op_node_2);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(control_flow_op_update_topological_order,
                  paddle::lite::mir::ControlFlowOpUpdateTopologicalOrderPass)
    .BindTargets({TARGET(kXPU)});
