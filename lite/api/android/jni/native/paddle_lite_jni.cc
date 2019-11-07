/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/api/android/jni/native/paddle_lite_jni.h"

#include <android/log.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/android/jni/native/convert_util_jni.h"
#include "lite/api/light_api.h"
#include "lite/api/paddle_api.h"

#include "model-encrypt-sdk/android/include/model_crypt.h"
#include "model-encrypt-sdk/android/include/model_decrypt.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace paddle {
namespace lite_api {

inline static std::shared_ptr<PaddlePredictor> *getPaddlePredictorPointer(
    JNIEnv *env, jobject jpaddle_predictor) {
  jclass jclazz = env->GetObjectClass(jpaddle_predictor);
  jfieldID jfield = env->GetFieldID(jclazz, "cppPaddlePredictorPointer", "J");
  jlong java_pointer = env->GetLongField(jpaddle_predictor, jfield);
  std::shared_ptr<PaddlePredictor> *ptr =
      reinterpret_cast<std::shared_ptr<PaddlePredictor> *>(java_pointer);
  return ptr;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_lite_PaddlePredictor_run(
    JNIEnv *env, jobject jpaddle_predictor) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return JNI_FALSE;
  }
  (*predictor)->Run();
  return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_lite_PaddlePredictor_getVersion(
    JNIEnv *env, jobject jpaddle_predictor) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return cpp_string_to_jstring(env, "");
  }
  return cpp_string_to_jstring(env, (*predictor)->GetVersion());
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_saveOptimizedModel(
    JNIEnv *env, jobject jpaddle_predictor, jstring model_dir) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return JNI_FALSE;
  }
  (*predictor)->SaveOptimizedModel(jstring_to_cpp_string(env, model_dir));
  return JNI_TRUE;
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getInputCppTensorPointer(
    JNIEnv *env, jobject jpaddle_predictor, jint offset) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return 0;
  }
  std::unique_ptr<Tensor> tensor =
      (*predictor)->GetInput(static_cast<int>(offset));
  std::unique_ptr<Tensor> *cpp_tensor_pointer =
      new std::unique_ptr<Tensor>(std::move(tensor));
  return reinterpret_cast<jlong>(cpp_tensor_pointer);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getOutputCppTensorPointer(
    JNIEnv *env, jobject jpaddle_predictor, jint offset) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return 0;
  }
  std::unique_ptr<const Tensor> tensor =
      (*predictor)->GetOutput(static_cast<int>(offset));
  std::unique_ptr<const Tensor> *cpp_tensor_pointer =
      new std::unique_ptr<const Tensor>(std::move(tensor));
  return reinterpret_cast<jlong>(cpp_tensor_pointer);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getCppTensorPointerByName(
    JNIEnv *env, jobject jpaddle_predictor, jstring name) {
  std::string cpp_name = jstring_to_cpp_string(env, name);
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return 0;
  }
  std::unique_ptr<const Tensor> tensor = (*predictor)->GetTensor(cpp_name);
  std::unique_ptr<const Tensor> *cpp_tensor_pointer =
      new std::unique_ptr<const Tensor>(std::move(tensor));
  return reinterpret_cast<jlong>(cpp_tensor_pointer);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_newCppPaddlePredictor__Lcom_baidu_\
paddle_lite_CxxConfig_2(JNIEnv *env,
                        jobject jpaddle_predictor,
                        jobject jcxxconfig) {
#ifndef LITE_ON_TINY_PUBLISH
  CxxConfig config = jcxxconfig_to_cpp_cxxconfig(env, jcxxconfig);
  std::shared_ptr<PaddlePredictor> predictor =
      paddle::lite_api::CreatePaddlePredictor(config);
  if (predictor == nullptr) {
    return 0;
  }
  std::shared_ptr<PaddlePredictor> *predictor_pointer =
      new std::shared_ptr<PaddlePredictor>(predictor);
  return reinterpret_cast<jlong>(predictor_pointer);
#else
  return 0;
#endif
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_newCppPaddlePredictor__Lcom_baidu_\
paddle_lite_MobileConfig_2(JNIEnv *env,
                           jobject jpaddle_predictor,
                           jobject jmobileconfig) {
  MobileConfig config = jmobileconfig_to_cpp_mobileconfig(env, jmobileconfig);

  unsigned char key[] = {0xee, 0x95, 0xec, 0xc7, 0x84, 0x98, 0xc4, 0x98,
                         0x84, 0xf6, 0x90, 0x91, 0xcd, 0xc7, 0x92, 0xf4,
                         0xe1, 0x8a, 0xf8, 0x8d, 0xe1, 0xcd, 0xdd, 0xe2,
                         0x97, 0xc1, 0x80, 0xe7, 0x96, 0xea};
  for (int i = 0; i < sizeof(key); i++) {
    key[i] ^= 0xAC;
  }
  __android_log_print(ANDROID_LOG_INFO, "ENC", "key:%s", key);

  void *context = NULL;
  unsigned int sign = 0;
  void *model_file_map = NULL;
  void *param_file_map = NULL;
  unsigned int model_file_size = 0;
  unsigned int param_file_size = 0;
  unsigned char *decrypt_model_output = NULL;
  unsigned char *decrypt_param_output = NULL;
  unsigned int decrypt_model_output_size = 0;
  unsigned int decrypt_param_output_size = 0;

  int ret;
  ret = init_crypt_context(key, sizeof(key), &context, &sign);
  if (0 != ret) {
    __android_log_print(
        ANDROID_LOG_INFO, "ENC", "failed to init_crypt_context");
    return -1;
  }
  __android_log_print(ANDROID_LOG_INFO, "test", "sign:%d", sign);

  auto model_dir = config.model_dir();
  const char *model_path = (model_dir + "/__model__.nb").c_str();
  __android_log_print(ANDROID_LOG_INFO, "ENC", "model_path:%s", model_path);
  ret = open_file_map(model_path, &model_file_map, &model_file_size);
  if (0 != ret) {
    __android_log_print(
        ANDROID_LOG_INFO, "ENC", "failed to open model_file_map");
    return -1;
  }
  const char *param_path = (model_dir + "/param.nb").c_str();
  __android_log_print(ANDROID_LOG_INFO, "ENC", "param_path:%s", param_path);
  ret = open_file_map(param_path, &param_file_map, &param_file_size);
  if (0 != ret) {
    __android_log_print(
        ANDROID_LOG_INFO, "ENC", "failed to open param_file_map");
    return -1;
  }
  __android_log_print(
      ANDROID_LOG_INFO, "test", "model_file_size:%d", model_file_size);
  __android_log_print(
      ANDROID_LOG_INFO, "test", "param_file_size:%d", param_file_size);

  ret = model_decrypt(context,
                      sign,
                      (unsigned char *)model_file_map,
                      model_file_size,
                      &decrypt_model_output,
                      &decrypt_model_output_size);
  if (0 != ret) {
    __android_log_print(ANDROID_LOG_INFO, "ENC", "failed to model_decrypt");
    return -1;
  }
  __android_log_print(ANDROID_LOG_INFO,
                      "ENC",
                      "decrypt_model_output_size:%d",
                      decrypt_model_output_size);
  ret = model_decrypt(context,
                      sign,
                      (unsigned char *)param_file_map,
                      param_file_size,
                      &decrypt_param_output,
                      &decrypt_param_output_size);
  if (0 != ret) {
    __android_log_print(ANDROID_LOG_INFO, "ENC", "failed to param_decrypt");
    return -1;
  }
  __android_log_print(ANDROID_LOG_INFO,
                      "ENC",
                      "decrypt_param_output_size:%d",
                      decrypt_param_output_size);

  close_file_map(model_file_map, model_file_size);
  close_file_map(param_file_map, param_file_size);
  model_file_map = NULL;
  param_file_map = NULL;
  uninit_crypt_context(context);
  context = NULL;

  config.set_model_buffer(reinterpret_cast<const char *>(decrypt_model_output),
                          decrypt_model_output_size,
                          reinterpret_cast<const char *>(decrypt_param_output),
                          decrypt_param_output_size);

  std::shared_ptr<PaddlePredictor> predictor =
      paddle::lite_api::CreatePaddlePredictor(config);
  if (predictor == nullptr) {
    return 0;
  }
  std::shared_ptr<PaddlePredictor> *predictor_pointer =
      new std::shared_ptr<PaddlePredictor>(predictor);
  return reinterpret_cast<jlong>(predictor_pointer);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_deleteCppPaddlePredictor(
    JNIEnv *env, jobject jpaddle_predictor, jlong java_pointer) {
  if (java_pointer == 0) {
    return JNI_FALSE;
  }
  std::shared_ptr<PaddlePredictor> *ptr =
      reinterpret_cast<std::shared_ptr<PaddlePredictor> *>(java_pointer);
  ptr->reset();
  delete ptr;
  return JNI_TRUE;
}

}  // namespace lite_api
}  // namespace paddle

#ifdef __cplusplus
}
#endif
