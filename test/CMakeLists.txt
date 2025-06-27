set(target ${TESTS})
find_package(MPI)

file(GLOB hdrs "*.h*")
file(GLOB srcs "*.cpp")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(${target} ${srcs} ${hdrs})
target_link_libraries(${target} gtest ${LIBRARY})
target_link_libraries(${target} ${MPI_C_LIBRARIES})
