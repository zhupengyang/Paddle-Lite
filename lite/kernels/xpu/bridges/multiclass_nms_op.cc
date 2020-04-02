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

#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int MulticlassNmsConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto boxes_name = op_info->Input("BBoxes").front();
  auto boxes = scope->FindTensor(boxes_name);

  auto scores_name = op_info->Input("Scores").front();
  auto scores = scope->FindTensor(scores_name);

  auto out_name = op_info->Output("Out").front();
  auto out= scope->FindMutableTensor(out_name);
  out->Resize({boxes->dims()[1], 6});

  int background_label = op_info->GetAttr<int>("background_label");
  int keep_top_k = op_info->GetAttr<int>("keep_top_k");
  int nms_top_k = op_info->GetAttr<int>("nms_top_k");
  float score_threshold = op_info->GetAttr<float>("score_threshold");
  float nms_threshold = op_info->GetAttr<float>("nms_threshold");
  float nms_eta = op_info->GetAttr<float>("nms_eta");
  bool normalized = op_info->HasAttr("normalized")
                        ? op_info->GetAttr<bool>("normalized")
                        : true;

  // boxes node
  std::shared_ptr<Node> boxes_node = nullptr;
  if (graph->Has(boxes_name)) {
    boxes_node = graph->Get(boxes_name);
  } else {
    boxes_node = graph->Add(boxes_name, *boxes);
  }

  // scores node
  std::shared_ptr<Node> scores_node = nullptr;
  if (graph->Has(scores_name)) {
    scores_node = graph->Get(scores_name);
  } else {
    scores_node = graph->Add(scores_name, *scores);
  }

  // multiclass_nms node
  graph->Add(out_name,
             graph->builder_.CreateMulticlassNMS(*boxes_node->data(),
                                           *scores_node->data(),
                                           score_threshold,
                                           nms_top_k,
                                           keep_top_k,
                                           nms_threshold,
                                           normalized,
                                           nms_eta,
                                           background_label));

  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(multiclass_nms,
                         kXPU,
                         paddle::lite::subgraph::xpu::MulticlassNmsConverter);
