cmake_minimum_required(VERSION 3.0)
project(BundleFusion)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fpermissive -pthread -lrt -fPIC")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(VISUALIZATION "Build BundleFusion With Pangolin" ON)

find_package(CUDA REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS -arch=sm_35;-std=c++14;-lm)
ADD_DEFINITIONS(-fpermissive)

find_package (Threads)
find_package (Eigen3 REQUIRED)
find_package (OpenCV REQUIRED)
list(APPEND CMAKE_INCLUDE_PATH "/usr/local/include")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)



file(GLOB_RECURSE cpp_file src/*.cpp)
file(GLOB_RECURSE cu_file src/*.cu)

if(VISUALIZATION)
  find_package(Pangolin 0.2 QUIET)
  message("-- found PANGOLIN, add visualization plug.")
  if(Pangolin_FOUND)
    find_package(Boost COMPONENTS system thread)
    add_definitions(-DWITH_VISUALIZATION)
    include_directories(
      ${Pangolin_INCLUDE_DIRS}
      ${Boost_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/include/Visualization/
    )
    
  endif()
else()
  message("-- not found PANGOLIN, visualization not enable.")
  file(GLOB_RECURSE Visualization_CPP_Files src/PangolinOutputWrapper.cpp)
  file(GLOB_RECURSE Visualization_CU_Files src/PangolinOutputWrapper.cu)
  list(REMOVE_ITEM cpp_file ${Visualization_CPP_Files})
  list(REMOVE_ITEM cu_file ${Visualization_CU_Files}) 
endif()

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}/
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/include/mLib/include
  ${PROJECT_SOURCE_DIR}/include/mLibExternal
  ${PROJECT_SOURCE_DIR}/include/cutil/cutil/inc/
  ${PROJECT_SOURCE_DIR}/include/FreeImage/
  ${PROJECT_SOURCE_DIR}/include/interface/
  ${PROJECT_SOURCE_DIR}/include/DepthSensing/
  ${PROJECT_SOURCE_DIR}/include/SiftGPU/
  ${PROJECT_SOURCE_DIR}/include/Solver/
)


cuda_add_library(BundleFusion  SHARED
  ${cu_file} 
  ${cpp_file}
)

target_link_libraries(BundleFusion 
  cuda 
  freeimage
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}

)

if(Pangolin_FOUND)
  target_link_libraries(BundleFusion
    ${Pangolin_LIBRARIES}
    boost_system 
    boost_thread
  )
endif()

# install(EXPORT gputsdfrecon-targets
#     NAMESPACE gputsdfrecon::
#     FILE gputsdfrecon-config.cmake
#     DESTINATION lib/cmake/gputsdfrecon)

add_executable(bundle_fusion_example example/bundle_fusion_example.cpp)
target_link_libraries(bundle_fusion_example 
   BundleFusion
)
# 
# include_directories(
# ${PROJECT_SOURCE_DIR}/example/include/)
# 
# add_executable(example_depthfusion example/example_depthfusion.cpp example/native.cpp)
# target_link_libraries(example_depthfusion 
#   gpu_tsdf_bf
#   )
#   
# add_executable(example_rkd_fusion example/example_rkd_fusion.cpp example/native.cpp)
# target_link_libraries(example_rkd_fusion gpu_tsdf_bf)
# 
# add_executable(example_rkd_fusion_cd example/example_rkd_fusion_cd.cpp example/native.cpp)
# 
# target_link_libraries(example_rkd_fusion_cd gpu_tsdf_bf)
  
#add_executable(example_tsdf_integrate_orbslam example/example_orbslam_tsdf_integrate.cpp)
#target_link_libraries(example_tsdf_integrate_orbslam 
#  gpu_tsdf_bf
#  )

