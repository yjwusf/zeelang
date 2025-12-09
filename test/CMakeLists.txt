configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ZEE_TEST_DEPENDS
  FileCheck count not
  zee-opt
)

add_lit_testsuite(check-zee "Running the Zee regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ZEE_TEST_DEPENDS}
)
set_target_properties(check-zee PROPERTIES FOLDER "Tests")

add_lit_testsuites(ZEE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ZEE_TEST_DEPENDS})
