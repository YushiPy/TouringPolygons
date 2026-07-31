cmake_minimum_required(VERSION 3.20)
project(tpp-convex)

set(CMAKE_CXX_STANDARD 26)

set(TARGET "main" CACHE STRING "Target to build")

file(GLOB_RECURSE SOURCES src/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX "src/main.*\\.cpp")

add_executable(${PROJECT_NAME} ${SOURCES} src/${TARGET}.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)

target_compile_options(${PROJECT_NAME} PRIVATE -O3)

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Gurobi
if(DEFINED GUROBI_HOME)
    set(GUROBI_HOME ${GUROBI_HOME})
elseif(DEFINED ENV{GUROBI_HOME})
    set(GUROBI_HOME $ENV{GUROBI_HOME})
else()
    message(FATAL_ERROR "GUROBI_HOME not set")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${GUROBI_HOME}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${GUROBI_HOME}/lib)

# Find the versioned gurobi lib (e.g. libgurobi110.so, libgurobi120.so)
file(GLOB GUROBI_LIB "${GUROBI_HOME}/lib/libgurobi[0-9]*.dylib")
list(FILTER GUROBI_LIB EXCLUDE REGEX ".*_light.*")
if(NOT GUROBI_LIB)
    message(FATAL_ERROR "Could not find versioned Gurobi shared library in ${GUROBI_HOME}/lib")
endif()
list(GET GUROBI_LIB 0 GUROBI_LIB)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${GUROBI_LIB}
    ${GUROBI_HOME}/lib/libgurobi_c++.a
)
