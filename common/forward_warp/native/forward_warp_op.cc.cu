// Taken from https://github.com/simonmeister/UnFlow/blob/master/ops/forward_warp_op.cu.cc.
// Commit bac9bbaf49be44b9e1c1f004fce4fb04b247763d.
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#define _USE_MATH_DEFINES
#include <cmath>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define gauss(x, y, std)

typedef Eigen::GpuDevice GPUDevice;

__global__ void ForwardWarpKernel(const int32 nthreads,
                                  const float* images, const float* flows,
                                  int batch, int height, int width, int channels,
                                  float variance, float* output) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = c + channels * (src_x + width * (src_y + height * b)).
    int idx = out_idx;
		const int c = idx % channels;
		idx /= channels;
		const int pixel_index = idx;
		const int flow_index = pixel_index * 2;
		const int src_x = idx % width;
		idx /= width;
		const int src_y = idx % height;
		const int b = idx / height;

    const float target_x = src_x + flows[flow_index];
    const float target_y = src_y + flows[flow_index + 1];

    const float std = sqrtf(variance);
    const float dist = std * 2.0;
    const int k = int(dist) + 2;

    // center pixel closest to mapping location.
#define IMG_OFFSET(iy, ix) (c + channels * (ix + width * (iy + height * b)))
    const float image_value = images[out_idx];
    const float x_m_k = target_x - k;
    const float x_p_k = target_x + k;
    const float y_m_k = target_y - k;
    const float y_p_k = target_y + k;
    const int floor_x_m_k = (int)floorf(x_m_k);
    const int floor_x_p_k = (int)floorf(x_p_k);
    const int floor_y_m_k = (int)floorf(y_m_k);
    const int floor_y_p_k = (int)floorf(y_p_k);
    if (floor_x_m_k < width && floor_x_p_k >= 0
        && floor_y_m_k < height && floor_y_p_k >= 0) {
      const int min_n_x = x_m_k > 0 ? floor_x_m_k : 0;
      const int min_n_y = y_m_k > 0 ? floor_y_m_k : 0;
      const int max_n_x = x_p_k < width? floor_x_p_k : width - 1;
      const int max_n_y = y_p_k < height? floor_y_p_k : height - 1;

      const float gauss_divisor = 2 * std * std;
      const float gauss_normalizer = gauss_divisor * float(M_PI);
      for (int n_x = min_n_x; n_x <= max_n_x; ++n_x) {
        for (int n_y = min_n_y; n_y <= max_n_y; ++n_y) {
          const float x = n_x - target_x;
          const float y = n_y - target_y;
          const float weight = expf(-(x * x + y * y) / gauss_divisor) / gauss_normalizer;
          CudaAtomicAdd(output + IMG_OFFSET(n_y, n_x), weight * image_value);
        }
      }
    }
#undef IMG_OFFSET
  }
}

__global__ void ForwardWarpGradKernel(const int32 nthreads,
                                      const float* input_grad, const float* images, const float* flows,
                                      int batch, int height, int width, int channels, float variance,
                                      float* output_image_grad, float* output_flow_grad) {
  CUDA_1D_KERNEL_LOOP(in_idx, nthreads) {
    // in_idx = c + channels * (src_x + width * (src_y + height * b)).
    int idx = in_idx;
		const int c = idx % channels;
		idx /= channels;
		const int pixel_index = idx;
		const int flow_index = pixel_index * 2;
		const int src_x = idx % width;
		idx /= width;
		const int src_y = idx % height;
		const int b = idx / height;

    const float target_x = src_x + flows[flow_index];
    const float target_y = src_y + flows[flow_index + 1];

    const float std = sqrtf(variance);
    const float dist = std * 2.0;
    const int k = int(dist) + 2;

    float du = 0.0;
    float dv = 0.0;

#define IMG_OFFSET(iy, ix) (c + channels * (ix + width * (iy + height * b)))
    float d_img = 0.0;
    const float image_value = images[in_idx];
    const float x_m_k = target_x - k;
    const float x_p_k = target_x + k;
    const float y_m_k = target_y - k;
    const float y_p_k = target_y + k;
    const int floor_x_m_k = (int)floorf(x_m_k);
    const int floor_x_p_k = (int)floorf(x_p_k);
    const int floor_y_m_k = (int)floorf(y_m_k);
    const int floor_y_p_k = (int)floorf(y_p_k);
    if (floor_x_m_k < width && floor_x_p_k >= 0
        && floor_y_m_k < height && floor_y_p_k >= 0) {
      const int min_n_x = x_m_k > 0? floor_x_m_k : 0;
      const int min_n_y = y_m_k > 0? floor_y_m_k : 0;
      const int max_n_x = x_p_k < width? floor_x_p_k : width - 1;
      const int max_n_y = y_p_k < height? floor_y_p_k : height - 1;

      const float gauss_divisor = 2 * std * std;
      const float gauss_normalizer = gauss_divisor * float(M_PI);
      for (int n_x = min_n_x; n_x <= max_n_x; ++n_x) {
        for (int n_y = min_n_y; n_y <= max_n_y; ++n_y) {
          const float x = n_x - target_x;
          const float y = n_y - target_y;
          const float weight = expf(-(x * x + y * y) / gauss_divisor) / gauss_normalizer;
          const float weighted_din = input_grad[IMG_OFFSET(n_y, n_x)] * weight;
          const float factor = 2 * weighted_din / gauss_divisor * image_value;
          du += factor * x;
          dv += factor * y;
          d_img += weighted_din;
        }
      }
    }

    output_image_grad[in_idx] = d_img;
    CudaAtomicAdd(output_flow_grad + flow_index, du);
    CudaAtomicAdd(output_flow_grad + flow_index + 1, dv);
  }
#undef IMG_OFFSET
}

void ForwardWarp(const GPUDevice& d,
                 typename TTypes<float, 4>::ConstTensor images,
                 typename TTypes<float, 4>::ConstTensor flows,
                 typename TTypes<float, 4>::Tensor output,
                 float variance) {
  const int batch = images.dimension(0);
  const int height = images.dimension(1);
  const int width = images.dimension(2);
  const int channels = images.dimension(3);

  const int total_count = batch * height * width * channels;
  if (total_count == 0) return;

  CudaLaunchConfig config;

  // Initialize output with all zeros.
  config = GetCudaLaunchConfig(total_count, d);
  SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, output.data());

  config = GetCudaLaunchConfig(total_count, d);
  ForwardWarpKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, images.data(), flows.data(),
      batch, height, width, channels,
      variance, output.data());
}

void ForwardWarpGrad(const GPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor input_grad,
                     typename TTypes<float, 4>::ConstTensor original_images,
                     typename TTypes<float, 4>::ConstTensor original_flows,
                     typename TTypes<float, 4>::Tensor output_image_grad,
                     typename TTypes<float, 4>::Tensor output_flow_grad,
                     float variance) {
  const int batch = input_grad.dimension(0);
  const int height = input_grad.dimension(1);
  const int width = input_grad.dimension(2);
  const int channels = input_grad.dimension(3);

  int total_count = batch * height * width * 2;
  if (total_count == 0) return;

  // Initialize output_flow_grad with all zeros.
  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
  SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, output_flow_grad.data());

  // Initialize output_image_grad with all zeros.
  total_count = batch * height * width * channels;
  config = GetCudaLaunchConfig(total_count, d);
  SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, output_image_grad.data());

  // Accumulate.
  config = GetCudaLaunchConfig(total_count, d);
  ForwardWarpGradKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, input_grad.data(),
      original_images.data(), original_flows.data(),
      batch, height, width, channels, variance,
      output_image_grad.data(), output_flow_grad.data());
}

#endif  // GOOGLE_CUDA
