tribits_get_package_enable_status(Kokkos  KokkosEnable "")

IF (KokkosEnable)
  MESSAGE("-- " "Skip adding flags for OpenMP because Kokkos flags does that ...")
  SET(OpenMP_CXX_FLAGS_OVERRIDE " ")
ENDIF()
