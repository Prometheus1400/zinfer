cmake_minimum_required(VERSION 3.25)

foreach(required MINIJINJA_SOURCE NATIVE_BUILD OUTPUT_LIBRARY JOBS)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing -D${required}")
  endif()
endforeach()

file(MAKE_DIRECTORY "${NATIVE_BUILD}")
file(LOCK "${NATIVE_BUILD}/.build.lock" GUARD PROCESS TIMEOUT 600)

find_program(CARGO_EXECUTABLE cargo
  HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin"
  REQUIRED)

# Use upstream's committed lockfile and emit a static C library without editing
# its cdylib manifest. Enable tojson for Hugging Face chat templates.
execute_process(
  COMMAND "${CARGO_EXECUTABLE}" rustc
    --manifest-path "${MINIJINJA_SOURCE}/Cargo.toml"
    --package minijinja-cabi --lib --crate-type staticlib
    --features minijinja/json
    --locked --release --target aarch64-apple-darwin
    --target-dir "${NATIVE_BUILD}/target"
    --jobs "${JOBS}"
  COMMAND_ERROR_IS_FATAL ANY
)

configure_file(
  "${NATIVE_BUILD}/target/aarch64-apple-darwin/release/libminijinja_cabi.a"
  "${OUTPUT_LIBRARY}"
  COPYONLY
)
