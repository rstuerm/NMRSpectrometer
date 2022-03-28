cmake_minimum_required(VERSION 3.21)
project(AD2_software_C)

set(CMAKE_CXX_STANDARD 14)

add_executable(AD2_software_C main.cpp)

if (UNIX AND NOT APPLE)
	target_link_libraries(AD2_software_C "/usr/lib/libdwf.so")
elseif (APPLE)
	find_library(DWF_LIBRARY dwf)
	target_link_libraries(AD2_software_C ${DWF_LIBRARY})
endif()