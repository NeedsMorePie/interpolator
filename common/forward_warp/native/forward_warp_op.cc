// Taken from https://github.com/simonmeister/UnFlow/blob/master/ops/forward_warp_op.cc.
// Commit bac9bbaf49be44b9e1c1f004fce4fb04b247763d.
#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

// TODO assert input flow channel count = 2, assert matching numbers in all other dims

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

void ForwardWarp(const GPUDevice& d,
                 typename TTypes<float, 4>::ConstTensor images,
                 typename TTypes<float, 4>::ConstTensor flows,
                 typename TTypes<float, 4>::Tensor output,
                 float variance);

void ForwardWarpGrad(const GPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor input_grad,
                     typename TTypes<float, 4>::ConstTensor original_images,
                     typename TTypes<float, 4>::ConstTensor original_flows,
                     typename TTypes<float, 4>::Tensor output_image_grad,
                     typename TTypes<float, 4>::Tensor output_flow_grad,
                     float variance);

class ForwardWarpOp : public OpKernel {
public:
  explicit ForwardWarpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("variance", &variance_));
    // Check that variance_ is positive
    OP_REQUIRES(context, variance_ >= 0.0f,
                errors::InvalidArgument("Need variance_ >= 0, got ", variance_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    const Tensor& flow = context->input(1);

    typename TTypes<float, 4>::ConstTensor image_data = image.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor flow_data = flow.tensor<float, 4>();

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(), &output));
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    ForwardWarp(context->eigen_device<GPUDevice>(),
                image_data, flow_data, output_data, variance_);
  }

private:
  float variance_;
};

class ForwardWarpOpGrad : public OpKernel {
public:
  explicit ForwardWarpOpGrad(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("variance", &variance_));
    // Check that variance_ is positive
    OP_REQUIRES(context, variance_ >= 0.0f,
                errors::InvalidArgument("Need variance_ >= 0, got ", variance_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& original_images = context->input(1);
    const Tensor& original_flows = context->input(2);

    Tensor* output_image_grads = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, original_images.shape(),
                                                     &output_image_grads));
    Tensor* output_flow_grads = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, original_flows.shape(),
                                                     &output_flow_grads));

    typename TTypes<float, 4>::ConstTensor input_data = input.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor original_images_data = original_images.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor original_flows_data = original_flows.tensor<float, 4>();
    typename TTypes<float, 4>::Tensor output_image_grads_data = output_image_grads->tensor<float, 4>();
    typename TTypes<float, 4>::Tensor output_flow_grads_data = output_flow_grads->tensor<float, 4>();

    ForwardWarpGrad(context->eigen_device<GPUDevice>(),
                    input_data, original_images_data, original_flows_data,
                    output_image_grads_data, output_flow_grads_data, variance_);
  }

private:
  float variance_;
};

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

REGISTER_OP("ForwardWarp")
  .Attr("variance: float = 1.0")
  .Input("images: float")
  .Input("flows: float")
  .Output("output: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    ShapeHandle in = c->input(0);
    DimensionHandle batch = c->Dim(in, 0);
    DimensionHandle height = c->Dim(in, 1);
    DimensionHandle width = c->Dim(in, 2);
    DimensionHandle channels = c->Dim(in, 3);
    c->set_output(0, c->MakeShape({batch, height, width, channels}));
    return Status::OK();
  });

REGISTER_OP("ForwardWarpGrad")
  .Attr("variance: float = 1.0")
  .Input("grads: float")
  .Input("original_images: float")
  .Input("original_flows: float")
  .Output("output_image_grad: float")
  .Output("output_flow_grad: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(1));
    c->set_output(1, c->input(2));
    return Status::OK();
  });

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("ForwardWarp").Device(DEVICE_GPU), ForwardWarpOp);
REGISTER_KERNEL_BUILDER(Name("ForwardWarpGrad").Device(DEVICE_GPU), ForwardWarpOpGrad);

#endif // GOOGLE_CUDA
