add_executable(nilm_posix
    detector.c
    main.c
)

target_include_directories(nilm_posix
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(nilm_posix PRIVATE c_std_99)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nilm_posix PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(nilm_posix PRIVATE m)
