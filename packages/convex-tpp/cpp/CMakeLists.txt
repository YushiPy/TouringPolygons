cmake_minimum_required(VERSION 3.20)
project(tpp-convex)

set(CMAKE_CXX_STANDARD 26)

set(TARGET "main" CACHE STRING "Target to build")
option(TPP_ENABLE_GUROBI "Build the optional Gurobi baseline solver" OFF)

if(APPLE)
    set(OpenMP_ROOT /opt/homebrew/opt/libomp)
    set(OpenMP_INCLUDE_DIR /opt/homebrew/opt/libomp/include)

    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")

    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")

    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
endif()

file(GLOB_RECURSE SOURCES src/*.cpp)
list(FILTER SOURCES EXCLUDE REGEX "src/main.*\\.cpp")

if(NOT TPP_ENABLE_GUROBI)
    list(FILTER SOURCES EXCLUDE REGEX "src/solvers/gurobi\\.cpp")
endif()

add_executable(${PROJECT_NAME} ${SOURCES} src/${TARGET}.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include)

target_compile_options(${PROJECT_NAME} PRIVATE -O3)

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

if(APPLE)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenMP_INCLUDE_DIR})
endif()

if(TPP_ENABLE_GUROBI)
    if(DEFINED GUROBI_HOME)
        set(GUROBI_HOME ${GUROBI_HOME})
    elseif(DEFINED ENV{GUROBI_HOME})
        set(GUROBI_HOME $ENV{GUROBI_HOME})
    else()
        message(FATAL_ERROR "GUROBI_HOME not set")
    endif()

    target_include_directories(${PROJECT_NAME} PRIVATE ${GUROBI_HOME}/include)
    target_link_directories(${PROJECT_NAME} PRIVATE ${GUROBI_HOME}/lib)

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
endif()
