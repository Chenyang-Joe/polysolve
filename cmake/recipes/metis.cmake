# METIS 5.1.0 — graph and matrix partitioning (Apache 2.0 since v5.2)
#
# We use whatever METIS the system / conda env provides via find_library +
# find_path. Reasons over CPM:
#   - METIS depends on GKlib which has its own CMake quirks, makes CPM hairy
#   - Ubuntu's libmetis-dev (5.1.0) is already at the right ABI for our needs
#   - polysolve doesn't otherwise need METIS, so a soft system dep is fine
#
# Provides imported target METIS::METIS that polysolve_linear can link against.

if(TARGET METIS::METIS)
    return()
endif()

# Search order:
#   1. CMAKE_PREFIX_PATH and conda env (CONDA_PREFIX)
#   2. Standard system locations (Ubuntu's /usr/lib/x86_64-linux-gnu)
find_path(METIS_INCLUDE_DIR
    NAMES metis.h
    HINTS ENV CONDA_PREFIX
    PATH_SUFFIXES include
)
find_library(METIS_LIBRARY
    NAMES metis
    HINTS ENV CONDA_PREFIX
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
    REQUIRED_VARS METIS_INCLUDE_DIR METIS_LIBRARY
)

if(METIS_FOUND)
    message(STATUS "Third-party: found METIS  inc=${METIS_INCLUDE_DIR}  lib=${METIS_LIBRARY}")
    add_library(METIS::METIS UNKNOWN IMPORTED)
    set_target_properties(METIS::METIS PROPERTIES
        IMPORTED_LOCATION             "${METIS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
