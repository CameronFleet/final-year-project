cmake_minimum_required(VERSION 3.12)
project(CAA)

set(CMAKE_CXX_STANDARD 14)
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

find_package(Qt5Widgets REQUIRED)

add_executable(CAA main.cpp ControlPanel.cpp ControlPanel.h)

target_link_libraries(CAA Qt5::Widgets)