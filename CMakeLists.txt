cmake_minimum_required(VERSION 3.3)
project(ml VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS "on")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(GGML_STANDALONE ON)

option(GGML_BUILD_TESTS "ml: build tests" ${GGML_STANDALONE})
option(GGML_CUBLAS                  "ggml: use cuBLAS"                   ON)

if(GGML_BUILD_TESTS)
    message(STATUS "ON")
endif()

if(NOT GGML_BUILD_TESTS)
    message(STATUS "OFF")
endif()

message(STATUS "${CMAKE_SYSTEM_NAME}")

# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     # 针对 GCC
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#     # 针对 Clang
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#     # 针对 Visual Studio
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
# endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
# set(CMAKE_C_STANDARD 11)
# set(CMAKE_C_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)


add_subdirectory(src)
add_subdirectory(common)
add_subdirectory(tests)
add_subdirectory(examples)

