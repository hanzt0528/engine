set(TARGET ml)
add_compile_options("-fpermissive")


if (GGML_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    set(CUDAToolkit_ROOT /usr/local/cuda/ )
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit)

    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_CUDA_SOURCES ml-cuda.cu ml-cuda.h)

        add_compile_definitions(GGML_USE_CUBLAS)

        if (GGML_STATIC)
            set(GGML_EXTRA_LIBS ${GGML_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(GGML_EXTRA_LIBS ${GGML_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

        if (CMAKE_BUILD_TYPE MATCHES Debug)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
        endif()
    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()



add_library(${TARGET}
    ml.cc
    ml-alloc.cc
    ml-backend.cc
    ml-impl.h
    ../include/ml/ml.h
    ${GGML_CUDA_SOURCES}
    # ${GGML_OPENCL_SOURCES}
    # ${GGML_METAL_SOURCES}
    )

target_include_directories(${TARGET} PUBLIC
    .
    ../include
    ../include/ml
    ${GGML_EXTRA_INCS}
    )

install(TARGETS ${TARGET}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include/ml
    )

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

#target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries(${TARGET} PUBLIC Threads::Threads  ${LLAMA_EXTRA_LIBS})

if (MSVC)
    target_link_libraries(${TARGET} PUBLIC ${GGML_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})
else()
    target_link_libraries(${TARGET} PUBLIC m Threads::Threads ${GGML_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${LLAMA_EXTRA_LIBS})
endif()
#target_compile_features(${TARGET} PRIVATE cxx_std_11)
