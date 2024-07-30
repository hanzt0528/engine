set(TARGET op_conv2d)
#add_executable(${TARGET} main.cpp ml.cpp)
add_executable(${TARGET} main.cc)

target_link_libraries(${TARGET} PRIVATE ml)

install(TARGETS ${TARGET} RUNTIME)
