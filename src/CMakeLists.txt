set(TARGET ml)
add_compile_options("-fpermissive")
add_library(${TARGET}
    ml.cc
    ml-alloc.cc
    ml-backend.cc
    ml-impl.h
    ../include/ml/ml.h

    # ${GGML_CUDA_SOURCES}
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
target_link_libraries(${TARGET} PUBLIC Threads::Threads  ${LLAMA_EXTRA_LIBS})

#target_compile_features(${TARGET} PRIVATE cxx_std_11)
