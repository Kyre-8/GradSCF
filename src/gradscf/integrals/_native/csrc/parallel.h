#pragma once
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

// Explicit opt-in: do not inherit a BLAS/OpenMP thread count and oversubscribe
// concurrent XLA callbacks. The caller owns its process-wide CPU budget.
inline int native_threads(int tasks) {
  const char* setting=std::getenv("GRADSCF_NATIVE_THREADS");
  if (!setting) return 1;
  int count=0;
  for (const char* p=setting; *p; ++p) {
    if (*p<'0' || *p>'9' || count>(std::numeric_limits<int>::max()-(*p-'0'))/10)
      throw std::invalid_argument("GRADSCF_NATIVE_THREADS must be a positive integer");
    count=count*10+(*p-'0');
  }
  if (count<1) throw std::invalid_argument("GRADSCF_NATIVE_THREADS must be a positive integer");
  return std::min(count,tasks);
}

// Mutable libcint tables and shell scratch belong to each worker. Only small
// parameter vectors are replicated; no integral tensor/Jacobian is cached.
// Strided outer shells spread the triangular three-center work across workers.
template<class Driver,class Apply>
void parallel_vjp(Driver& driver,double* output,Apply apply) {
  const int workers=native_threads(driver.active);
  if (workers==1) {apply(driver,output);return;}
  std::vector<std::vector<double>> partial(workers);
  std::vector<std::exception_ptr> errors(workers);
  std::vector<std::thread> threads;
  auto run=[&](int worker) {
    try {
      Driver local=driver;
      local.first_shell=worker;local.shell_stride=workers;
      partial[worker].assign(driver.count,0.);
      apply(local,partial[worker].data());
    } catch (...) {errors[worker]=std::current_exception();}
  };
  try {
    for (int w=1;w<workers;++w) threads.emplace_back(run,w);
  } catch (...) {
    for (auto& thread:threads) thread.join();
    throw;
  }
  run(0);
  for (auto& thread:threads) thread.join();
  for (const auto& error:errors) if (error) std::rethrow_exception(error);
  // Fixed reduction order for reproducibility at a fixed worker count.
  for (const auto& values:partial)
    for (size_t p=0;p<driver.count;++p) output[p]+=values[p];
}
