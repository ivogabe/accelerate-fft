#include <pthread.h>
#include <inttypes.h>
#include <accelerate_llvm_native_ffi.h>
#include <fftw3.h>

// Plan construction is not thread-safe in fftw. Hence we use this lock to
// protect plan construction.
static pthread_mutex_t planner_lock = PTHREAD_MUTEX_INITIALIZER;

struct CtxDim1 {
  struct AccelerateKernelHeader header;
  int64_t sign; // 1 or -1
  int64_t n0;
  void *input; // float or double
  int64_t n0_;
  void *output;
};
struct CtxDim2 {
  struct AccelerateKernelHeader header;
  int64_t sign;
  int64_t n0;
  int64_t n1;
  void *input; // float or double
  int64_t n0_;
  int64_t n1_;
  void *output;
};
struct CtxDim3 {
  struct AccelerateKernelHeader header;
  int64_t sign;
  int64_t n0;
  int64_t n1;
  int64_t n2;
  void *input; // float or double
  int64_t n0_;
  int64_t n1_;
  int64_t n2_;
  void *output;
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
uint8_t accelerate_fft_dim1_double(struct CtxDim1 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) {
    // Initialization or finalization
    return 0;
  }

  pthread_mutex_lock(&planner_lock);
  // If we do not set FFTW_ESTIMATE, fftw will already overwite the output
  // array (and thus the input array, if the transformation is in-place).
  fftw_plan p = fftw_plan_dft_1d((int) ctx->n0, (fftw_complex*) ctx->input, (fftw_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy(p);
}
uint8_t accelerate_fft_dim1_float(struct CtxDim1 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) { return 0; }

  pthread_mutex_lock(&planner_lock);
  fftwf_plan p = fftwf_plan_dft_1d((int) ctx->n0, (fftwf_complex*) ctx->input, (fftwf_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy_float(p);
}

// Dim2
uint8_t accelerate_fft_dim2_double(struct CtxDim2 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) { return 0; }

  pthread_mutex_lock(&planner_lock);
  fftw_plan p = fftw_plan_dft_2d((int) ctx->n0, (int) ctx->n1, (fftw_complex*) ctx->input, (fftw_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy(p);
}
uint8_t accelerate_fft_dim2_float(struct CtxDim2 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) { return 0; }

  pthread_mutex_lock(&planner_lock);
  fftwf_plan p = fftwf_plan_dft_2d((int) ctx->n0, (int) ctx->n1, (fftwf_complex*) ctx->input, (fftwf_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy_float(p);
}

// Dim3
uint8_t accelerate_fft_dim3_double(struct CtxDim3 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) { return 0; }

  pthread_mutex_lock(&planner_lock);
  fftw_plan p = fftw_plan_dft_3d((int) ctx->n0, (int) ctx->n1, (int) ctx->n2, (fftw_complex*) ctx->input, (fftw_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy(p);
}
uint8_t accelerate_fft_dim3_float(struct CtxDim3 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) { return 0; }

  pthread_mutex_lock(&planner_lock);
  fftwf_plan p = fftwf_plan_dft_3d((int) ctx->n0, (int) ctx->n1, (int) ctx->n2, (fftwf_complex*) ctx->input, (fftwf_complex*) ctx->output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  return execute_and_destroy_float(p);
}

// Multi-dimensional, only over the inner dimension (i.e. per row)
uint8_t accelerate_fft_many_double(struct CtxDim2 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) { return 0; }

  if (ctx->n0 == 0) {
    return 0;
  } 

  // The first row of the input will be properly aligned, since Accelerate
  // allocates arrays with a high alignment. Depending on the size of rows,
  // later rows may or may not be aligned. Since all rows have the same size,
  // we can check that by checking the alignment of the second row (if it
  // exists; if there is no second row we don't care about this).
  
  // Construct plan with pointers to the second row if it exists, otherwise
  // the first row, to provide fftw with info on alignment.
  // The plan is later executed separately per row.
  fftw_complex *plan_input;
  fftw_complex *plan_output;
  if (ctx->n0 == 1) {
    plan_input = (fftw_complex*) ctx->input;
    plan_output = (fftw_complex*) ctx->output;
  } else {
    plan_input = &((fftw_complex*) ctx->input)[ctx->n1];
    plan_output = &((fftw_complex*) ctx->output)[ctx->n1];
  }
  pthread_mutex_lock(&planner_lock);
  fftw_plan p = fftw_plan_dft_1d((int) ctx->n1, plan_input, plan_output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  // TODO: Make this a parallel for loop, via Work Assisting
  for (int64_t row = 0; row < ctx->n0; row++) {
    fftw_execute_dft(p, &((fftw_complex*) ctx->input)[ctx->n1 * row], &((fftw_complex*) ctx->output)[ctx->n1 * row]);
  }

  pthread_mutex_lock(&planner_lock);
  fftw_destroy_plan(p);
  pthread_mutex_unlock(&planner_lock);
  return 0;
}
uint8_t accelerate_fft_many_float(struct CtxDim2 *ctx, uint8_t *locks, uint32_t thread_idx, uint32_t max_threads) {
  if (thread_idx >= 0xFFFFFFFE) { return 0; }
  if (ctx->n0 == 0) { return 0; } 

  fftwf_complex *plan_input;
  fftwf_complex *plan_output;
  if (ctx->n0 == 1) {
    plan_input = (fftwf_complex*) ctx->input;
    plan_output = (fftwf_complex*) ctx->output;
  } else {
    plan_input = &((fftwf_complex*) ctx->input)[ctx->n1];
    plan_output = &((fftwf_complex*) ctx->output)[ctx->n1];
  }
  pthread_mutex_lock(&planner_lock);
  fftwf_plan p = fftwf_plan_dft_1d((int) ctx->n1, plan_input, plan_output, (int) ctx->sign, FFTW_ESTIMATE);
  pthread_mutex_unlock(&planner_lock);

  for (int64_t row = 0; row < ctx->n0; row++) {
    fftwf_execute_dft(p, &((fftwf_complex*) ctx->input)[ctx->n1 * row], &((fftwf_complex*) ctx->output)[ctx->n1 * row]);
  }

  pthread_mutex_lock(&planner_lock);
  fftwf_destroy_plan(p);
  pthread_mutex_unlock(&planner_lock);
  return 0;
}
