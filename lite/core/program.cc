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

#include "lite/core/program.h"
#include <algorithm>
#include <map>
#include "lite/model_parser/cpp/block_desc.h"
#include "lite/model_parser/cpp/op_desc.h"
#include "lite/model_parser/cpp/var_desc.h"
#include "lite/operators/conditional_block_op.h"
#include "lite/operators/subgraph_op.h"
#include "lite/operators/while_op.h"
#ifdef LITE_WITH_PRECISION_PROFILE
#include "lite/core/profile/precision_profiler.h"
#endif

namespace paddle {
namespace lite {

void RuntimeProgram::SaveOpInfosToProgram(cpp::ProgramDesc* program_desc) {
  CHECK(program_desc);
  // NOTE: RuntimeProgram do not has all meta info, so save model just update
  // upon origin model
  CHECK(program_desc->BlocksSize());
  int block_idx = 0;
  auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  block_desc->ClearOps();
  for (auto& node : instructions_) {
    auto op_info = *node.op()->op_info();
    auto op_type = op_info.Type();
    if (op_type == "subgraph" && op_info.GetAttr<int32_t>("sub_block") == 0) {
      // It's a new subgraph op when its sub_block_idx = 0, Now we add its
      // subblock desc to the program desc, Then update its sub_block_idx to
      // the index of block desc of the program desc.
      auto subgraph_op = const_cast<operators::SubgraphOp*>(
          static_cast<const operators::SubgraphOp*>(node.op()));
      auto* sub_program_desc = subgraph_op->GetProgramDesc();
      CHECK(sub_program_desc);
      auto* sub_block_desc = program_desc->AddBlock<cpp::BlockDesc>();
      *sub_block_desc = *sub_program_desc->GetBlock<cpp::BlockDesc>(0);
      sub_block_desc->SetParentIdx(block_idx);
      delete sub_program_desc;
      subgraph_op->SetProgramDesc(program_desc);
      op_info.SetAttr<int32_t>("sub_block", program_desc->BlocksSize() - 1);
      auto* scope = subgraph_op->scope();
      // Attach op and kernel again to update the new block_idx and
      // program_desc
      subgraph_op->Attach(op_info, scope);
      subgraph_op->AttachKernel(node.mutable_kernel());
      // Update main block desc after a new subblock desc is added
      block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
    }
    auto* op_desc = block_desc->AddOp<cpp::OpDesc>();
    *op_desc = op_info;
    op_desc->SetAttr(kKernelTypeAttr, node.kernel()->SerializedKernelType());
  }
}

// `UpdateVarsOfProgram` will remove unused var_descs and add new created
// vars' descs in the block 0. Now, the type of a new created var can only
// be LOD_TENSOR.
void RuntimeProgram::UpdateVarsOfProgram(cpp::ProgramDesc* program_desc) {
  int block_idx = 0;
  CHECK(program_desc);
  CHECK(program_desc->BlocksSize());
  std::map<std::string, cpp::VarDesc> origin_var_maps;
  auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  auto var_size = block_desc->VarsSize();
  for (size_t i = 0; i < var_size; i++) {
    auto v = block_desc->GetVar<cpp::VarDesc>(i);
    auto name = v->Name();
    origin_var_maps.emplace(name, *v);
  }

  block_desc->ClearVars();
  for (auto& node : instructions_) {
    auto* op = const_cast<lite::OpLite*>(node.op());
    auto* kernel = node.kernel();
    auto* scope = op->scope();
    auto in_names = op->op_info()->input_names();
    auto out_names = op->op_info()->output_names();

    std::vector<std::string> all_names(in_names);
    all_names.insert(all_names.end(), out_names.begin(), out_names.end());
    for (auto& name : all_names) {
      auto it = origin_var_maps.find(name);
      if (it != origin_var_maps.end()) {
        auto* v = block_desc->AddVar<cpp::VarDesc>();
        v->SetName((it->second).Name());
        v->SetType((it->second).GetType());
        v->SetPersistable((it->second).Persistable());
      } else {
        // New created vars must be LOD_TENSOR
        auto* v = block_desc->AddVar<cpp::VarDesc>();
        v->SetName(name);
        std::string arg_name;
        const Type* type = nullptr;
        if (op->op_info()->GetInputArgname(name, &arg_name)) {
          type = kernel->GetInputDeclType(arg_name);
        } else if (op->op_info()->GetOutputArgname(name, &arg_name)) {
          type = kernel->GetOutputDeclType(arg_name);
        } else {
          LOG(FATAL) << "not find " << name;
        }
        if (type->IsTensor()) {
          v->SetType(cpp::VarDesc::Type::LOD_TENSOR);
          auto tensor = scope->FindVar(name)->GetMutable<Tensor>();
          v->SetPersistable(tensor->persistable());
        } else if (type->IsTensorList()) {
          // Set persistable=false for tensor array
          v->SetType(cpp::VarDesc::Type::LOD_TENSOR_ARRAY);
          v->SetPersistable(false);
        } else {
          CHECK(false) << "unsupported var type";
        }
      }
    }
  }
}

// Create runtime program from sub_block desc according to block_idx and
// program_desc, which is used for while/conditional_block/subgraph op.
RuntimeProgram::RuntimeProgram(int block_idx,
                               cpp::ProgramDesc* program_desc,
                               Scope* exec_scope)
    : exec_scope_(exec_scope) {
  CHECK(block_idx >= 0 && block_idx < program_desc->BlocksSize());
  auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  for (size_t op_idx = 0; op_idx < block_desc->OpsSize(); op_idx++) {
    auto* op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    // Create op and pick up the best kernel
    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "no Op found for " << op_type;
    op->Attach(*op_desc, exec_scope_);
    std::unique_ptr<KernelBase> picked_kernel;
    if (op_desc->HasAttr(kKernelTypeAttr)) {
      // Create op and pick up the best kernel according to the
      // kKernelTypeAttr attribute
      auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
      std::string alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      VLOG(3) << "Found the attr '" << kKernelTypeAttr << "': " << kernel_type
              << " for " << op_type;
      auto kernels = op->CreateKernels({place});
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      auto it = std::find_if(
          kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& it) {
            return it->alias() == alias;
          });
      CHECK(it != kernels.end());
      picked_kernel = std::move(*it);
    } else {
      // TODO(hong19860320) add kernel picking according to the type of input
      // and output tensors
      VLOG(3) << "The attr '" << kKernelTypeAttr
              << "' not found, pick the first kernel for " << op_type;
      std::vector<std::unique_ptr<KernelBase>> kernels;
#if defined(LITE_WITH_ARM)
      kernels = op->CreateKernels({Place{TARGET(kARM)}, Place{TARGET(kHost)}});
#elif defined(LITE_WITH_X86)
      kernels = op->CreateKernels({Place{TARGET(kX86)}, Place{TARGET(kHost)}});
#endif
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      picked_kernel = std::move(kernels.front());
    }
    picked_kernel->SetContext(
        ContextScheduler::Global().NewContext(picked_kernel->target()));
    instructions_.emplace_back(std::move(op), std::move(picked_kernel));
  }
  CHECK(!instructions_.empty()) << "no instructions";
#ifdef LITE_WITH_PROFILE
  set_profiler();
#endif
}

void RuntimeProgram::Run() {
#ifdef LITE_WITH_PRECISION_PROFILE
  auto inst_precision_profiler = paddle::lite::profile::PrecisionProfiler();
  std::string precision_profiler_summary =
      inst_precision_profiler.GetSummaryHeader();
#endif

  for (auto& inst : instructions_) {
#ifndef LITE_WITH_FPGA
    if (inst.is_feed_fetch_op()) continue;
#endif
#ifdef LITE_WITH_CUDA
    if (inst.need_sync()) {
      inst.Sync();
    }
#endif
    inst.Run();
#ifdef LITE_WITH_PRECISION_PROFILE
#ifndef LITE_WITH_FPGA
    precision_profiler_summary +=
        inst_precision_profiler.GetInstPrecision(&inst);
#endif
#endif  // LITE_WITH_PRECISION_PROFILE
  }
#ifdef LITE_WITH_PROFILE
  LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kDispatch, false, 1);
#endif
#ifdef LITE_WITH_PRECISION_PROFILE
  LOG(INFO) << "\n" << precision_profiler_summary;
#endif
}

void Program::Build(const cpp::ProgramDesc& program_desc) {
  CHECK(ops_.empty()) << "Executor duplicate Build found";

  // Create operators.
  auto program = program_desc;
  CHECK(program.BlocksSize());
  auto& main_block = *program.GetBlock<cpp::BlockDesc>(0);
  for (size_t i = 0; i < main_block.OpsSize(); ++i) {
    auto& op_desc = *main_block.GetOp<cpp::OpDesc>(i);
    auto op_type = op_desc.Type();
    VLOG(4) << "create Op [" << op_type << "]";
    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "no Op found for " << op_type;
    if (op_type == "while" || op_type == "conditional_block" ||
        op_type == "subgraph") {
      auto sub_block_idx = op_desc.GetAttr<int32_t>("sub_block");
      CHECK(sub_block_idx >= 0 && sub_block_idx < program.BlocksSize())
          << "Invalid attribute sub_block(" << sub_block_idx << ") for "
          << op_type;
      auto sub_block_desc = const_cast<cpp::ProgramDesc&>(program_desc)
                                .GetBlock<cpp::BlockDesc>(sub_block_idx);
      CHECK(sub_block_desc);
      if (op_type == "while") {
        static_cast<operators::WhileOpLite*>(op.get())->SetSubBlock(
            sub_block_desc);
      } else if (op_type == "conditional_block") {
        static_cast<operators::ConditionalBlockOpLite*>(op.get())->SetSubBlock(
            sub_block_desc);
      } else if (op_type == "subgraph") {
        static_cast<operators::SubgraphOp*>(op.get())->SetProgramDesc(
            const_cast<cpp::ProgramDesc*>(&program_desc));
      }
    }
    ops_.emplace_back(std::move(op));
    ops_.back()->Attach(op_desc, exec_scope_);
  }
}

void Program::PrepareWorkspace(const cpp::ProgramDesc& prog,
                               const std::vector<std::string>& var_names) {
  CHECK(!exec_scope_) << "Duplicate PrepareWorkspace found";
  exec_scope_ = &scope_->NewScope();
  // Create Feed and Fetch var.
  scope_->Var("feed")->GetMutable<std::vector<lite::Tensor>>();
  scope_->Var("fetch")->GetMutable<std::vector<lite::Tensor>>();
  tmp_vars_.push_back("feed");
  tmp_vars_.push_back("fetch");

  auto VarPrecision2KernlPrecision =
      [](const lite::VarDescAPI::Type& type) -> PrecisionType {
    switch (type) {
      case lite::VarDescAPI::Type::FP32:
        return PRECISION(kFloat);
      case lite::VarDescAPI::Type::FP16:
        return PRECISION(kFP16);
      case lite::VarDescAPI::Type::INT8:
        return PRECISION(kInt8);
      case lite::VarDescAPI::Type::INT16:
        return PRECISION(kInt16);
      case lite::VarDescAPI::Type::INT32:
        return PRECISION(kInt32);
      case lite::VarDescAPI::Type::INT64:
        return PRECISION(kInt64);
      default:
        // LOG(FATAL) << "not supported type: " << static_cast<int>(type);
        return PRECISION(kUnk);
    }
  };

  auto program = prog;
  CHECK(program.BlocksSize());
  for (size_t b = 0; b < program.BlocksSize(); ++b) {
    auto& main_block = *program.GetBlock<cpp::BlockDesc>(b);
    for (size_t i = 0; i < main_block.VarsSize(); ++i) {
      auto& var_desc = *main_block.GetVar<cpp::VarDesc>(i);
      if (!var_desc.Persistable()) {
        if (var_desc.GetType() == lite::VarDescAPI::Type::LOD_TENSOR &&
            VarPrecision2KernlPrecision(var_desc.GetDataType()) !=
                PRECISION(kUnk)) {
          var_data_type_[var_desc.Name()] =
              VarPrecision2KernlPrecision(var_desc.GetDataType());
        }
        tmp_vars_.push_back(var_desc.Name());
        VLOG(4) << "var name: " << var_desc.Name() << " type is "
                << static_cast<int>(var_desc.GetType()) << " data type is "
                << static_cast<int>(var_desc.GetDataType());
        exec_scope_->Var(var_desc.Name());
        if (b > 0) {
          VLOG(4) << "var: " << var_desc.Name();
        }
      } else {
        if (var_desc.Name() == "feed" || var_desc.Name() == "fetch") continue;
        weights_.push_back(var_desc.Name());
        if (var_desc.Persistable()) scope_->Var(var_desc.Name());
      }
    }
  }

  for (auto i : var_names) {
    exec_scope_->LocalVar(i);
    auto* tensor = scope_->Var(i)->GetMutable<lite::Tensor>();
    auto* sub_tensor = exec_scope_->Var(i)->GetMutable<lite::Tensor>();
    sub_tensor->CopyDataFrom(*tensor);
  }
}

void Instruction::Run() {
#ifdef LITE_WITH_PROFILE
  CHECK(profiler_) << "Profiler pointer of kernel can not be nullptr. "
                      "When LITE_WITH_PROFILE is defined, please set a "
                      "Profiler for Instruction.";
  profiler_->StartTiming(
      profile::Type::kCreate, profile_id_, kernel_->mutable_context());
#endif
  CHECK(op_) << "op null";
  CHECK(kernel_) << "kernel null";

  if (first_epoch_) {
    first_epoch_ = false;
    CHECK(op_->CheckShape());
  }

  if (op_->run_once() && has_run_) {
    return;
  }

  op_->InferShape();
  kernel_->Launch();
  has_run_ = true;

#ifdef LITE_WITH_PROFILE
  if (first_epoch_for_profiler_) {
    kernel_->SetIsKernelTest(false);
    SetProfileRuntimeOpInfo(profiler_->GetOpCharacter(profile_id_));
    first_epoch_for_profiler_ = false;
  }
#endif
}

STL::ostream& operator<<(STL::ostream& os, const Instruction& other) {
  os << other.kernel_->summary() << "\t(" << other.kernel_->doc() << ")";
  return os;
}

}  // namespace lite
}  // namespace paddle
