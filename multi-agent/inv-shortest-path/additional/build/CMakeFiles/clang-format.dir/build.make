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

# Utility rule file for clang-format.

# Include any custom commands dependencies for this target.
include CMakeFiles/clang-format.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/clang-format.dir/progress.make

CMakeFiles/clang-format:
	clang-format -i /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/a_star.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/a_star_epsilon.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/assignment.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/cbs.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/cbs_ta.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/ecbs.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/ecbs_ta.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/neighbor.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/next_best_assignment.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/planresult.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/include/libMultiRobotPlanning/sipp.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/a_star.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/a_star_epsilon.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/assignment.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/cbs.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/cbs_ta.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/ecbs.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/ecbs_ta.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/mapf_prioritized_sipp.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/next_best_assignment.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/shortest_path_heuristic.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/shortest_path_heuristic.hpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/sipp.cpp /Users/yonathan/Documents/Internships/KURF\ 2021/libMultiRobotPlanning-master/example/timer.hpp

clang-format: CMakeFiles/clang-format
clang-format: CMakeFiles/clang-format.dir/build.make
.PHONY : clang-format

# Rule to build all files generated by this target.
CMakeFiles/clang-format.dir/build: clang-format
.PHONY : CMakeFiles/clang-format.dir/build

CMakeFiles/clang-format.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clang-format.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clang-format.dir/clean

CMakeFiles/clang-format.dir/depend:
	cd "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build" "/Users/yonathan/Documents/Internships/KURF 2021/libMultiRobotPlanning-master/build/CMakeFiles/clang-format.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/clang-format.dir/depend

