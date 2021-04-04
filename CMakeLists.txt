cmake_minimum_required(VERSION 3.17)
project(CeresSolverPractise)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g3 -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -O3")


find_package(Ceres REQUIRED)

set(PCL_DIR "/home/goktug/libraries/libpcl-1.11.1/share/pcl-1.11/")
find_package(PCL 1.11.1 REQUIRED )

add_executable(Point2PointTransEstimation
        Point2PointTransEstimation.cpp
        )

add_executable(CurveFitting
        CurveFitting.cpp
        )


include_directories(
        ${CERES_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_link_libraries(
        Point2PointTransEstimation
        Ceres::ceres
        ${PCL_LIBRARIES}
       )

target_link_libraries(
        CurveFitting
        Ceres::ceres
        ${PCL_LIBRARIES}
)