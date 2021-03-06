# CMakeLists inspired by Tpetra/src/CMakeLists.txt

INCLUDE(TrilinosCreateClientTemplateHeaders)

#
# A) Package-specific configuration options
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.hpp)

#
# SRC
#

SET(HEADERS "")
APPEND_GLOB(HEADERS */*.hpp)
APPEND_GLOB(HEADERS */*.cpp)

SET(SOURCE mac.cpp) # Avoid empty library (MacOs 'ar' utility doesn't work on empty library).

#
#
#

SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR}/Matrix)
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR}/StridedMap)
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR}/Cloner)
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR}/Utils)

TRILINOS_CREATE_CLIENT_TEMPLATE_HEADERS(${DIR})

# Must glob the binary dir last to get all of the auto-generated headers
SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_BINARY_DIR})

#
# Define the targets for package's library(s)
#

TRIBITS_ADD_LIBRARY(
  xpetra-sup
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS xpetra
  )
