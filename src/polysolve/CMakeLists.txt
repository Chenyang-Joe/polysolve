set(SOURCES
	Utils.hpp
	Utils.cpp
	save_problem.hpp
	save_problem.cpp
	JSONUtils.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve_linear PRIVATE ${SOURCES})


