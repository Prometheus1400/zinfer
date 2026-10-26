cmake_minimum_required(VERSION 3.25)

foreach(required MLX_C_SOURCE NATIVE_BUILD METAL JOBS)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing -D${required}")
  endif()
endforeach()

# Editor background checks and terminal builds may run at the same time.
# Ninja and CMake cannot safely share a writable build tree without a lock.
file(MAKE_DIRECTORY "${NATIVE_BUILD}")
file(LOCK "${NATIVE_BUILD}/.build.lock" GUARD PROCESS TIMEOUT 600)

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${MLX_C_SOURCE}" -B "${NATIVE_BUILD}" -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DMLX_C_BUILD_EXAMPLES=OFF
    -DMLX_C_USE_SYSTEM_MLX=OFF
    -DMLX_BUILD_CPU=ON
    "-DMLX_BUILD_METAL=${METAL}"
    -DMLX_METAL_JIT=OFF
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    "-DCMAKE_INSTALL_RPATH=@loader_path"
    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${NATIVE_BUILD}/lib"
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${NATIVE_BUILD}" --parallel "${JOBS}"
  COMMAND_ERROR_IS_FATAL ANY
)
