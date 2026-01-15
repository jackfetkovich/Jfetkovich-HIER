file(REMOVE_RECURSE
  "libmpac_go2_local.a"
  "libmpac_go2_local.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/mpac_go2_local.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
