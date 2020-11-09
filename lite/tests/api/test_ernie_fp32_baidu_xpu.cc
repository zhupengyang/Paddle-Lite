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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/lite_api_test_helper.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/tests/api/bert_utility.h"
#include "lite/utils/cp_logging.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 100, "iteration times to run");

namespace paddle {
namespace lite {

template <class T = int64_t>
void ReadRank12RawData(const std::string& input_data_dir,
                       std::vector<std::vector<T>>* input0,
                       std::vector<std::vector<T>>* input1,
                       std::vector<std::vector<T>>* input2,
                       std::vector<std::vector<float>>* input3,
                       std::vector<std::vector<int64_t>>* input_shapes) {
  auto lines = ReadLines(input_data_dir);
  for (auto line : lines) {
    std::vector<std::string> shape_and_data = Split(line, ";");
    std::vector<int64_t> input_shape =
        Split<int64_t>(Split(shape_and_data[0], ",,")[1], " ");
    input_shapes->emplace_back(input_shape);

    std::vector<T> input0_data =
        Split<T>(Split(shape_and_data[0], ",,")[2], " ");
    input0->emplace_back(input0_data);
    std::vector<T> input1_data =
        Split<T>(Split(shape_and_data[1], ",,")[2], " ");
    input1->emplace_back(input1_data);
    std::vector<T> input2_data =
        Split<T>(Split(shape_and_data[2], ",,")[2], " ");
    input2->emplace_back(input2_data);
    std::vector<float> input3_data =
        Split<float>(Split(shape_and_data[3], ",,")[2], " ");
    input3->emplace_back(input3_data);
  }
}

float CalRank12OutAccuracy(const std::vector<float>& out,
                           const std::string& out_file) {
  auto lines = ReadLines(out_file);
  std::vector<float> ref_out;
  for (auto line : lines) {
    ref_out.emplace_back(std::stof(line));
  }

  int right_num = 0;
  for (size_t i = 0; i < out.size(); i++) {
    right_num += (std::fabs(out[i] - ref_out[i]) < 0.01f);
  }

  return static_cast<float>(right_num) / static_cast<float>(out.size());
}

template <typename T>
lite::Tensor GetTensorWithShape(std::vector<int64_t> shape) {
  lite::Tensor ret;
  ret.Resize(shape);
  T* ptr = ret.mutable_data<T>();
  for (int i = 0; i < ret.numel(); ++i) {
    ptr[i] = (T)1;
  }
  return ret;
}

TEST(Ernie, test_ernie_fp32_baidu_xpu) {
  lite_api::CxxConfig config;
  config.set_model_dir(FLAGS_model_dir);
  config.set_valid_places({lite_api::Place{TARGET(kXPU), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kX86), PRECISION(kFloat)},
                           lite_api::Place{TARGET(kHost), PRECISION(kFloat)}});
  config.set_xpu_workspace_l3_size_per_thread();
  auto predictor = lite_api::CreatePaddlePredictor(config);

  std::string input_data_file =
      FLAGS_data_dir + std::string("/ras_ernie_L12_input_data.init");
  std::vector<std::vector<int64_t>> input0;
  std::vector<std::vector<int64_t>> input1;
  std::vector<std::vector<int64_t>> input2;
  std::vector<std::vector<float>> input3;
  std::vector<std::vector<int64_t>> input_shapes;
  ReadRank12RawData(
      input_data_file, &input0, &input1, &input2, &input3, &input_shapes);

  for (int i = 0; i < FLAGS_warmup; ++i) {
    std::vector<int64_t> shape = {1, 64, 1};
    std::vector<int64_t> fill_value(64, 0);
    for (int j = 0; j < 4; j++) {
      FillTensor(predictor, j, shape, fill_value);
    }
    predictor->Run();
  }

  std::vector<float> out_rets;
  // out_rets.resize(FLAGS_iteration);
  double cost_time = 0;
  for (int i = 0; i < FLAGS_iteration; ++i) {
    FillTensor(predictor, 0, input_shapes[i], input0[i]);
    FillTensor(predictor, 1, input_shapes[i], input1[i]);
    FillTensor(predictor, 2, input_shapes[i], input2[i]);
    FillTensor(predictor, 3, input_shapes[i], input3[i]);

    double start = GetCurrentUS();
    predictor->Run();
    cost_time += GetCurrentUS() - start;

    auto output_tensor = predictor->GetOutput(0);
    auto output_shape = output_tensor->shape();
    auto output_data = output_tensor->data<float>();
    ASSERT_EQ(output_shape.size(), 2UL);
    ASSERT_EQ(output_shape[0], input_shapes[i][0]);
    ASSERT_EQ(output_shape[1], 1);

    size_t output_size = output_shape[0] * output_shape[1];
    out_rets.resize(out_rets.size() + output_size);
    memcpy(&(out_rets.at(out_rets.size() - output_size)),
           output_data,
           sizeof(float) * output_size);
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup
            << ", iteration: " << FLAGS_iteration << ", spend "
            << cost_time / FLAGS_iteration / 1000.0 << " ms in average.";

  std::string ref_out_file =
      FLAGS_data_dir + std::string("/ras_ernie_L12_output_data");
  float out_accuracy = CalRank12OutAccuracy(out_rets, ref_out_file);
  LOG(INFO) << "--- out_accuracy: " << out_accuracy;
  ASSERT_GT(out_accuracy, 0.95f);
}

}  // namespace lite
}  // namespace paddle
