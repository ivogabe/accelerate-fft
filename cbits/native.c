#include <inttypes.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>
#include <accelerate_llvm_native_ffi.h>
#include <fftw3.h>

// Plan construction is not thread-safe in fftw. Hence we use this lock to
// protect plan construction.
static pthread_mutex_t planner_lock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;

struct ParallelLoop {
  void *(*work)(char *);
  char *jobdata;
  size_t elsize;
};
static void parallel_loop_work(void* ctx, int64_t i) {
  struct ParallelLoop *loop = (struct ParallelLoop*) ctx;
  loop->work(loop->jobdata + loop->elsize * i);
}
static void parallel_loop(void *(*work)(char *), char *jobdata, size_t elsize, int njobs, void *data) {
  struct ParallelLoop loop;
  loop.work = work;
  loop.jobdata = jobdata;
  loop.elsize = elsize;
  accelerate_llvm_native_ffi_parallel_loop(parallel_loop_work, (void*) &loop, njobs);
}

static void initialize_fftw_threads() {
  initialized = true;
  fftw_init_threads();
  fftwf_init_threads();
  fftw_threads_set_callback(&parallel_loop, NULL);
  fftwf_threads_set_callback(&parallel_loop, NULL);
}

struct CtxDim1Double {
  struct AccelerateKernelHeader header;
  // Input & output
  int64_t sign; // 1 or -1
  int64_t n0;
  fftw_complex *input; // float or double
  int64_t n0_;
  fftw_complex *output;
  // Kernel memory: store the plan
  fftw_plan plan;
};
struct CtxDim1Float {
  struct AccelerateKernelHeader header;
  // Input & output
  int64_t sign; // 1 or -1
  int64_t n0;
  fftwf_complex *input; // float or double
  int64_t n0_;
  fftwf_complex *output;
  // Kernel memory: store the plan
  fftwf_plan plan;
};
struct CtxDim2Double {
  struct AccelerateKernelHeader header;
  // Input & output
  int64_t sign;
  int64_t n0;
  int64_t n1;
  fftw_complex *input; // float or double
  int64_t n0_;
  int64_t n1_;
  fftw_complex *output;
  // Kernel memory: store the plan
  fftw_plan plan;
};
struct CtxDim2Float {
  struct AccelerateKernelHeader header;
  // Input & output
  int64_t sign;
  int64_t n0;
  int64_t n1;
  fftwf_complex *input; // float or double
  int64_t n0_;
  int64_t n1_;
  fftwf_complex *output;
  // Kernel memory: store the plan
  fftwf_plan plan;
};
struct CtxDim3Double {
  struct AccelerateKernelHeader header;
  // Input & output
  int64_t sign;
  int64_t n0;
  int64_t n1;
  int64_t n2;
  fftw_complex *input; // float or double
  int64_t n0_;
  int64_t n1_;
  int64_t n2_;
  fftw_complex *output;
  // Kernel memory: store the plan
  fftw_plan plan;
};
struct CtxDim3Float {
  struct AccelerateKernelHeader header;
  // Input & output
  int64_t sign;
  int64_t n0;
  int64_t n1;
  int64_t n2;
  fftwf_complex *input; // float or double
  int64_t n0_;
  int64_t n1_;
  int64_t n2_;
  fftwf_complex *output;
  // Kernel memory: store the plan
  fftwf_plan plan;
};

inline uint8_t execute_and_destroy(fftw_plan p) {
  fftw_execute(p);
  pthread_mutex_lock(&planner_lock);
  fftw_destroy_plan(p);
  pthread_mutex_unlock(&planner_lock);
  return 0;
}
inline uint8_t execute_and_destroy_float(fftwf_plan p) {
  fftwf_execute(p);
  pthread_mutex_lock(&planner_lock);
  fftwf_destroy_plan(p);
  pthread_mutex_unlock(&planner_lock);
  return 0;
}

// Dim1
uint8_t accelerate_fft_dim1_double(struct CtxDim1Double *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  // This function is called three times: for initialization, parallel work
  // and finalization. Since fftw does not directly fit in this scheme for
  // parallelization, we perform all work during initialization.
  // Parallelism can still be exploited via
  // accelerate_llvm_native_ffi_parallel_loop.
  if (thread_idx != 0xFFFFFFFF) {
    return 0;
  }

  pthread_mutex_lock(&planner_lock);
  if (!initialized) initialize_fftw_threads();
  fftw_plan_with_nthreads(max_threads);
  // If we do not set FFTW_ESTIMATE, fftw will already overwite the output
  // array (and thus the input array, if the transformation is in-place).
  fftw_plan p = fftw_plan_dft_1d((int) ctx->n0, ctx->input, ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy(p);
}
uint8_t accelerate_fft_dim1_float(struct CtxDim1Float *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx != 0xFFFFFFFF) { return 0; }

  pthread_mutex_lock(&planner_lock);
  if (!initialized) initialize_fftw_threads();
  fftwf_plan_with_nthreads(max_threads);
  fftwf_plan p = fftwf_plan_dft_1d((int) ctx->n0, (fftwf_complex*) ctx->input, (fftwf_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy_float(p);
}

// Dim2
uint8_t accelerate_fft_dim2_double(struct CtxDim2Double *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx != 0xFFFFFFFF) { return 0; }

  pthread_mutex_lock(&planner_lock);
  if (!initialized) initialize_fftw_threads();
  fftw_plan_with_nthreads(max_threads);
  fftw_plan p = fftw_plan_dft_2d((int) ctx->n0, (int) ctx->n1, ctx->input, ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy(p);
}
uint8_t accelerate_fft_dim2_float(struct CtxDim2Float *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx != 0xFFFFFFFF) { return 0; }

  pthread_mutex_lock(&planner_lock);
  if (!initialized) initialize_fftw_threads();
  fftwf_plan_with_nthreads(max_threads);
  fftwf_plan p = fftwf_plan_dft_2d((int) ctx->n0, (int) ctx->n1, (fftwf_complex*) ctx->input, (fftwf_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy_float(p);
}

// Dim3
uint8_t accelerate_fft_dim3_double(struct CtxDim3Double *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx != 0xFFFFFFFF) { return 0; }

  pthread_mutex_lock(&planner_lock);
  if (!initialized) initialize_fftw_threads();
  fftw_plan_with_nthreads(max_threads);
  fftw_plan p = fftw_plan_dft_3d((int) ctx->n0, (int) ctx->n1, (int) ctx->n2, ctx->input, ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy(p);
}
uint8_t accelerate_fft_dim3_float(struct CtxDim3Float *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx != 0xFFFFFFFF) { return 0; }

  pthread_mutex_lock(&planner_lock);
  if (!initialized) initialize_fftw_threads();
  fftwf_plan_with_nthreads(max_threads);
  fftwf_plan p = fftwf_plan_dft_3d((int) ctx->n0, (int) ctx->n1, (int) ctx->n2, (fftwf_complex*) ctx->input, (fftwf_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy_float(p);
}

// Multi-dimensional, only over the inner dimension (i.e. per row)
uint8_t accelerate_fft_many_double(struct CtxDim2Double *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  // On empty inputs, we don't need to do any work for initialization, finalization or regular work.
  if (ctx->n0 == 0) {
    return 0;
  }

  if (thread_idx == 0xFFFFFFFF) {
    // Initialization

    // Alignment of rows:
    // The first row of the input will be properly aligned, since Accelerate
    // allocates arrays with a high alignment. Depending on the size of rows,
    // later rows may or may not be aligned. Since all rows have the same size,
    // we can check that by checking the alignment of the second row (if it
    // exists; if there is no second row we don't care about this).
    
    // Construct plan with pointers to the second row if it exists, otherwise
    // the first row, to provide fftw with info on alignment.
    // The plan is later executed separately per row.

    // We create a parallel plan (like in accelerate_fft_dim1_double) when ctx->n0 == 1,
    // otherwise we exploit the parallelism between rows.

    fftw_complex *plan_input;
    fftw_complex *plan_output;
    if (ctx->n0 == 1) {
      plan_input = ctx->input;
      plan_output = ctx->output;
    } else {
      plan_input = &ctx->input[ctx->n1];
      plan_output = &ctx->output[ctx->n1];
    }
    pthread_mutex_lock(&planner_lock);
    if (!initialized) initialize_fftw_threads();
    fftw_plan_with_nthreads(ctx->n0 == 1 ? max_threads : 1);
    ctx->plan = fftw_plan_dft_1d((int) ctx->n1, plan_input, plan_output, (int) ctx->sign, FFTW_ESTIMATE);
    pthread_mutex_unlock(&planner_lock);

    return ctx->n0 == 1 ? 1 : 0; // 1 means parallel; use multiple worker threads
  }
  if (thread_idx >= 0xFFFFFFFE) {
    // Finalization
    pthread_mutex_lock(&planner_lock);
    fftw_destroy_plan(ctx->plan);
    pthread_mutex_unlock(&planner_lock);
    return 0;
  }

  while (true) {
    int64_t row = atomic_fetch_add_explicit(&ctx->header.work_index, 1, memory_order_relaxed);
    if (row >= ctx->n0) break;

    fftw_execute_dft(ctx->plan, &ctx->input[ctx->n1 * row], &ctx->output[ctx->n1 * row]);
  }
  return 0;
}
uint8_t accelerate_fft_many_float(struct CtxDim2Float *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (ctx->n0 == 0) { return 0; } 

  if (thread_idx == 0xFFFFFFFF) {
    // Initialization
    fftwf_complex *plan_input;
    fftwf_complex *plan_output;
    if (ctx->n0 == 1) {
      plan_input = ctx->input;
      plan_output = ctx->output;
    } else {
      plan_input = &ctx->input[ctx->n1];
      plan_output = &ctx->output[ctx->n1];
    }
    pthread_mutex_lock(&planner_lock);
    if (!initialized) initialize_fftw_threads();
    fftwf_plan_with_nthreads(ctx->n0 == 1 ? max_threads : 1);
    ctx->plan = fftwf_plan_dft_1d((int) ctx->n1, plan_input, plan_output, (int) ctx->sign, FFTW_ESTIMATE);
    pthread_mutex_unlock(&planner_lock);

    return ctx->n0 == 1 ? 1 : 0;
  }
  if (thread_idx >= 0xFFFFFFFE) {
    // Finalization
    pthread_mutex_lock(&planner_lock);
    fftwf_destroy_plan(ctx->plan);
    pthread_mutex_unlock(&planner_lock);
    return 0;
  }

  while (true) {
    int64_t row = atomic_fetch_add_explicit(&ctx->header.work_index, 1, memory_order_relaxed);
    if (row >= ctx->n0) break;

    fftwf_execute_dft(ctx->plan, &ctx->input[ctx->n1 * row], &ctx->output[ctx->n1 * row]);
  }
  return 0;
}
