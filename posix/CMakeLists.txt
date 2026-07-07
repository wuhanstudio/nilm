CMake_minimum_required(VERSION 3.10)

project(nilm_posix C)

add_executable(nilm_posix
    detector.c
    main.c
    lime-tm/clause.c
    lime-tm/tsetlin.c
)

target_include_directories(nilm_posix
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/lime-tm
)

target_compile_definitions(nilm_posix
    PRIVATE
        TSETLIN_USING_STATIC_MODEL
)

target_compile_features(nilm_posix PRIVATE c_std_99)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nilm_posix PRIVATE -Wall -Wextra)
endif()

target_link_libraries(nilm_posix PRIVATE m)
