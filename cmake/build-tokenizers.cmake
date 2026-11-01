cmake_minimum_required(VERSION 3.25)

foreach(required TOKENIZERS_SOURCE TOKENIZERS_LOCK NATIVE_BUILD OUTPUT_LIBRARY JOBS)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing -D${required}")
  endif()
endforeach()

file(MAKE_DIRECTORY "${NATIVE_BUILD}")
file(LOCK "${NATIVE_BUILD}/.build.lock" GUARD PROCESS TIMEOUT 600)

find_program(CARGO_EXECUTABLE cargo
  HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin"
  REQUIRED)

# Build from a cache-local copy so Cargo leaves the pinned submodule untouched.
# The project lockfile pins the upstream crate's otherwise unlocked dependencies.
set(CRATE_SOURCE "${NATIVE_BUILD}/source")
file(MAKE_DIRECTORY "${CRATE_SOURCE}/src")
configure_file("${TOKENIZERS_SOURCE}/rust/Cargo.toml" "${CRATE_SOURCE}/Cargo.toml" COPYONLY)
configure_file("${TOKENIZERS_SOURCE}/rust/src/lib.rs" "${CRATE_SOURCE}/src/lib.rs" COPYONLY)
configure_file("${TOKENIZERS_LOCK}" "${CRATE_SOURCE}/Cargo.lock" COPYONLY)

execute_process(
  COMMAND "${CARGO_EXECUTABLE}" build
    --manifest-path "${CRATE_SOURCE}/Cargo.toml"
    --locked --release --target aarch64-apple-darwin
    --target-dir "${NATIVE_BUILD}/target"
    --jobs "${JOBS}"
  COMMAND_ERROR_IS_FATAL ANY
)

configure_file(
  "${NATIVE_BUILD}/target/aarch64-apple-darwin/release/libtokenizers_c.a"
  "${OUTPUT_LIBRARY}"
  COPYONLY
)
