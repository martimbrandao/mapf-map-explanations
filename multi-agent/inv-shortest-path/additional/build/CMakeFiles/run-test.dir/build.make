# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.20.5/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.20.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build"

# Utility rule file for run-test.

# Include any custom commands dependencies for this target.
include CMakeFiles/run-test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/run-test.dir/progress.make

CMakeFiles/run-test:
	python3 -m unittest discover -s /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/test

run-test: CMakeFiles/run-test
run-test: CMakeFiles/run-test.dir/build.make
.PHONY : run-test

# Rule to build all files generated by this target.
CMakeFiles/run-test.dir/build: run-test
.PHONY : CMakeFiles/run-test.dir/build

CMakeFiles/run-test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run-test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run-test.dir/clean

CMakeFiles/run-test.dir/depend:
	cd "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build/CMakeFiles/run-test.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/run-test.dir/depend

