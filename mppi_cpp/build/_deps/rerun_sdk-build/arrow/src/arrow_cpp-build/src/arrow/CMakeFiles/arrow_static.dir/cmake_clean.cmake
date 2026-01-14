file(REMOVE_RECURSE
  "../../release/libarrow.a"
  "../../release/libarrow.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CXX)
  include(CMakeFiles/arrow_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
