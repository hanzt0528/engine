# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /data/hanzt1/tools/cmake-3.29.1-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /data/hanzt1/tools/cmake-3.29.1-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/hanzt1/he/codes/engine/tests/llvm/LACommenter

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/build

# Include any dependencies generated for this target.
include CMakeFiles/LACommenter.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/LACommenter.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/LACommenter.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LACommenter.dir/flags.make

CMakeFiles/LACommenter.dir/LACommenter.cpp.o: CMakeFiles/LACommenter.dir/flags.make
CMakeFiles/LACommenter.dir/LACommenter.cpp.o: /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/LACommenter.cpp
CMakeFiles/LACommenter.dir/LACommenter.cpp.o: CMakeFiles/LACommenter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/data/hanzt1/he/codes/engine/tests/llvm/LACommenter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LACommenter.dir/LACommenter.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LACommenter.dir/LACommenter.cpp.o -MF CMakeFiles/LACommenter.dir/LACommenter.cpp.o.d -o CMakeFiles/LACommenter.dir/LACommenter.cpp.o -c /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/LACommenter.cpp

CMakeFiles/LACommenter.dir/LACommenter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/LACommenter.dir/LACommenter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/LACommenter.cpp > CMakeFiles/LACommenter.dir/LACommenter.cpp.i

CMakeFiles/LACommenter.dir/LACommenter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/LACommenter.dir/LACommenter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/LACommenter.cpp -o CMakeFiles/LACommenter.dir/LACommenter.cpp.s

# Object files for target LACommenter
LACommenter_OBJECTS = \
"CMakeFiles/LACommenter.dir/LACommenter.cpp.o"

# External object files for target LACommenter
LACommenter_EXTERNAL_OBJECTS =

libLACommenter.so: CMakeFiles/LACommenter.dir/LACommenter.cpp.o
libLACommenter.so: CMakeFiles/LACommenter.dir/build.make
libLACommenter.so: CMakeFiles/LACommenter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/data/hanzt1/he/codes/engine/tests/llvm/LACommenter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libLACommenter.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LACommenter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LACommenter.dir/build: libLACommenter.so
.PHONY : CMakeFiles/LACommenter.dir/build

CMakeFiles/LACommenter.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LACommenter.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LACommenter.dir/clean

CMakeFiles/LACommenter.dir/depend:
	cd /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/hanzt1/he/codes/engine/tests/llvm/LACommenter /data/hanzt1/he/codes/engine/tests/llvm/LACommenter /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/build /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/build /data/hanzt1/he/codes/engine/tests/llvm/LACommenter/build/CMakeFiles/LACommenter.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/LACommenter.dir/depend

