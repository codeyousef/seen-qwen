cmake_minimum_required(VERSION 3.25)
project(seen_qwen_cuda_reference LANGUAGES C CXX CUDA)

if(NOT DEFINED SEEN_CUDA_SOURCE_ROOT)
  message(FATAL_ERROR "SEEN_CUDA_SOURCE_ROOT must identify the packaged Seen CUDA runtime")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  message(FATAL_ERROR "FEL-1432 supports Linux x86-64 only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "Explicit CUDA architectures")
find_package(CUDAToolkit 12.0 REQUIRED)
set(SEEN_ENABLE_CUDA ON CACHE BOOL "Explicit opt-in to packaged Seen CUDA")
add_subdirectory("${SEEN_CUDA_SOURCE_ROOT}" seen_cuda_build)

add_library(seen_qwen_cuda SHARED src/reference_primitives.cu)
target_include_directories(seen_qwen_cuda PUBLIC include)
target_link_libraries(seen_qwen_cuda PUBLIC seen_cuda CUDA::cudart CUDA::cuda_driver)
target_compile_options(seen_qwen_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(seen_qwen_cuda PROPERTIES
  CXX_VISIBILITY_PRESET default
  CUDA_VISIBILITY_PRESET default
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME seen_qwen_cuda
  BUILD_RPATH "$ORIGIN")

add_executable(qwn_040a_cuda_test tests/reference_primitives_test.cu)
target_link_libraries(qwn_040a_cuda_test PRIVATE seen_qwen_cuda seen_cuda CUDA::cudart)
target_compile_options(qwn_040a_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(qwn_040a_cuda_test PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/seen_cuda_build")

add_executable(qwn_040a_header_contract tests/header_contract.c)
target_include_directories(qwn_040a_header_contract PRIVATE
  include "${SEEN_CUDA_SOURCE_ROOT}/include")
target_compile_options(qwn_040a_header_contract PRIVATE -Wall -Wextra -Werror)

add_executable(qwn_040b_cuda_test tests/reference_utilities_test.cu)
target_link_libraries(qwn_040b_cuda_test PRIVATE seen_qwen_cuda seen_cuda CUDA::cudart)
target_compile_options(qwn_040b_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(qwn_040b_cuda_test PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/seen_cuda_build")

add_executable(qwn_041a_cuda_test tests/gdn_causal_conv_test.cu)
target_link_libraries(qwn_041a_cuda_test PRIVATE seen_qwen_cuda seen_cuda CUDA::cudart)
target_compile_options(qwn_041a_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(qwn_041a_cuda_test PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/seen_cuda_build")

add_executable(qwn_041b_cuda_test tests/gdn_recurrent_decode_test.cu)
target_link_libraries(qwn_041b_cuda_test PRIVATE seen_qwen_cuda seen_cuda CUDA::cudart)
target_compile_options(qwn_041b_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(qwn_041b_cuda_test PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/seen_cuda_build")

add_executable(qwn_041c_cuda_test tests/gdn_recurrent_prefill_test.cu)
target_link_libraries(qwn_041c_cuda_test PRIVATE seen_qwen_cuda seen_cuda CUDA::cudart)
target_compile_options(qwn_041c_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(qwn_041c_cuda_test PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/seen_cuda_build")

add_executable(qwn_041d_cuda_test tests/gdn_gated_output_test.cu)
target_link_libraries(qwn_041d_cuda_test PRIVATE seen_qwen_cuda seen_cuda CUDA::cudart)
target_compile_options(qwn_041d_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(qwn_041d_cuda_test PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/seen_cuda_build")

add_executable(qwn_042a_cuda_test tests/attention_qk_geometry_test.cu)
target_link_libraries(qwn_042a_cuda_test PRIVATE seen_qwen_cuda seen_cuda CUDA::cudart)
target_compile_options(qwn_042a_cuda_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--threads=1;--Werror=cross-execution-space-call>)
set_target_properties(qwn_042a_cuda_test PROPERTIES BUILD_RPATH "$ORIGIN;$ORIGIN/seen_cuda_build")
