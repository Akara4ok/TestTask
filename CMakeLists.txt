cmake_minimum_required(VERSION 3.2)

project(TestTask)

find_package( OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(TestTask 
src/main.cpp
)

target_link_libraries(TestTask ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})