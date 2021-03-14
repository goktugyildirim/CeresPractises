cmake_minimum_required(VERSION 3.17)
project(ceres_practise)

set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)

add_executable(curve_fitting curve_fitting.cpp)

include_directories(
        ${CERES_INCLUDE_DIRS}
)

target_link_libraries(curve_fitting
        Ceres::ceres
)