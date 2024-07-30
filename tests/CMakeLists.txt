set(TARGET-1 test-mul-mat)
add_executable(${TARGET-1} test-mul-mat.cc )
target_link_libraries(${TARGET-1} PRIVATE ml)
install(TARGETS  ${TARGET-1} RUNTIME)

set(TARGET-2 test-conv2d)
add_executable(${TARGET-2} test-conv2d.cc )
target_link_libraries(${TARGET-2} PRIVATE ml)


install(TARGETS  ${TARGET-2} RUNTIME)
