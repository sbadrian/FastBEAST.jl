using TestItemRunner

@run_package_tests filter=ti->(:fast in ti.tags) verbose=true