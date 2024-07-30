set(TARGET lenet)
#add_executable(${TARGET} main.cpp ml.cpp)
add_executable(${TARGET} main.cc)

target_link_libraries(${TARGET} PRIVATE ml)

install(TARGETS ${TARGET} RUNTIME)

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

#target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PUBLIC Threads::Threads  ${LLAMA_EXTRA_LIBS})

target_compile_features(${TARGET} PRIVATE cxx_std_11)
