#include "svd_metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <dispatch/dispatch.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastpls_svd {
namespace {

id<MTLDevice> default_device() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

id<MTLCommandQueue> default_command_queue() {
  static id<MTLCommandQueue> queue = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = default_device();
    if (device != nil) {
      queue = [device newCommandQueue];
    }
  });
  return queue;
}

void copy_arma_to_mps_buffer(const arma::mat& X,
                             id<MTLBuffer> buffer,
                             const NSUInteger row_bytes) {
  const arma::uword nr = X.n_rows;
  const arma::uword nc = X.n_cols;
  char* base = static_cast<char*>([buffer contents]);

  for (arma::uword i = 0; i < nr; ++i) {
    float* row = reinterpret_cast<float*>(base + static_cast<NSUInteger>(i) * row_bytes);
    for (arma::uword j = 0; j < nc; ++j) {
      row[j] = static_cast<float>(X(i, j));
    }
  }
}

arma::mat copy_mps_buffer_to_arma(id<MTLBuffer> buffer,
                                  const arma::uword nr,
                                  const arma::uword nc,
                                  const NSUInteger row_bytes) {
  arma::mat out(nr, nc);
  const char* base = static_cast<const char*>([buffer contents]);

  for (arma::uword i = 0; i < nr; ++i) {
    const float* row = reinterpret_cast<const float*>(base + static_cast<NSUInteger>(i) * row_bytes);
    for (arma::uword j = 0; j < nc; ++j) {
      out(i, j) = static_cast<double>(row[j]);
    }
  }

  return out;
}

struct MetalMatrix {
  id<MTLBuffer> buffer;
  NSUInteger rows;
  NSUInteger cols;
  NSUInteger row_bytes;
};

MetalMatrix make_matrix(id<MTLDevice> device,
                        const NSUInteger rows,
                        const NSUInteger cols) {
  MetalMatrix M;
  M.rows = rows;
  M.cols = cols;
  M.row_bytes = [MPSMatrixDescriptor rowBytesFromColumns:cols dataType:MPSDataTypeFloat32];
  M.buffer = [device newBufferWithLength:(M.row_bytes * rows) options:MTLResourceStorageModeShared];
  if (M.buffer == nil) {
    throw std::runtime_error("metal_simpls_resident: failed to allocate Metal buffer");
  }
  std::fill_n(static_cast<char*>([M.buffer contents]), static_cast<std::size_t>(M.row_bytes * rows), 0);
  return M;
}

MetalMatrix make_matrix_from_arma(id<MTLDevice> device, const arma::mat& X) {
  MetalMatrix M = make_matrix(device, static_cast<NSUInteger>(X.n_rows), static_cast<NSUInteger>(X.n_cols));
  copy_arma_to_mps_buffer(X, M.buffer, M.row_bytes);
  return M;
}

MPSMatrix* mps_matrix(const MetalMatrix& M) {
  MPSMatrixDescriptor* desc = [MPSMatrixDescriptor
    matrixDescriptorWithRows:M.rows
    columns:M.cols
    rowBytes:M.row_bytes
    dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:M.buffer descriptor:desc];
}

MetalMatrix metal_matmul(id<MTLDevice> device,
                         id<MTLCommandQueue> queue,
                         const MetalMatrix& A,
                         const MetalMatrix& B,
                         const bool transpose_left,
                         const bool transpose_right) {
  const NSUInteger a_rows = transpose_left ? A.cols : A.rows;
  const NSUInteger a_cols = transpose_left ? A.rows : A.cols;
  const NSUInteger b_rows = transpose_right ? B.cols : B.rows;
  const NSUInteger b_cols = transpose_right ? B.rows : B.cols;
  if (a_cols != b_rows) {
    throw std::runtime_error("metal_simpls_resident: non-conformable resident matrices");
  }

  MetalMatrix C = make_matrix(device, a_rows, b_cols);
  MPSMatrix* a_matrix = mps_matrix(A);
  MPSMatrix* b_matrix = mps_matrix(B);
  MPSMatrix* c_matrix = mps_matrix(C);
  MPSMatrixMultiplication* mult = [[MPSMatrixMultiplication alloc]
    initWithDevice:device
    transposeLeft:transpose_left
    transposeRight:transpose_right
    resultRows:a_rows
    resultColumns:b_cols
    interiorColumns:a_cols
    alpha:1.0
    beta:0.0];

  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  if (command_buffer == nil) {
    throw std::runtime_error("metal_simpls_resident: failed to create command buffer");
  }
  [mult encodeToCommandBuffer:command_buffer leftMatrix:a_matrix rightMatrix:b_matrix resultMatrix:c_matrix];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  NSError* err = [command_buffer error];
  if (err != nil) {
    NSString* msg = [err localizedDescription];
    throw std::runtime_error(std::string("metal_simpls_resident: Metal command failed: ") +
                             std::string([msg UTF8String]));
  }
  return C;
}

arma::vec copy_metal_vector(const MetalMatrix& M) {
  const NSUInteger stride = M.row_bytes / sizeof(float);
  const float* base = static_cast<const float*>([M.buffer contents]);
  if (M.cols == 1) {
    arma::vec out(M.rows);
    for (NSUInteger i = 0; i < M.rows; ++i) out[i] = static_cast<double>(base[i * stride]);
    return out;
  }
  if (M.rows == 1) {
    arma::vec out(M.cols);
    for (NSUInteger j = 0; j < M.cols; ++j) out[j] = static_cast<double>(base[j]);
    return out;
  }
  throw std::runtime_error("metal_simpls_resident: expected a vector buffer");
}

void write_metal_col_vector(MetalMatrix& M, const arma::vec& x) {
  if (M.cols != 1 || M.rows != x.n_elem) {
    throw std::runtime_error("metal_simpls_resident: non-conformable vector write");
  }
  const NSUInteger stride = M.row_bytes / sizeof(float);
  float* base = static_cast<float*>([M.buffer contents]);
  for (NSUInteger i = 0; i < M.rows; ++i) base[i * stride] = static_cast<float>(x[i]);
}

id<MTLComputePipelineState> rank1_sub_pipeline(id<MTLDevice> device) {
  static id<MTLComputePipelineState> pipeline = nil;
  static dispatch_once_t once;
  static std::string error_message;
  dispatch_once(&once, ^{
    NSString* source =
      @"#include <metal_stdlib>\n"
       "using namespace metal;\n"
       "kernel void rank1_sub(device float* S [[buffer(0)]],\n"
       "                      device const float* v [[buffer(1)]],\n"
       "                      device const float* row [[buffer(2)]],\n"
       "                      constant uint& nrow [[buffer(3)]],\n"
       "                      constant uint& ncol [[buffer(4)]],\n"
       "                      constant uint& s_stride [[buffer(5)]],\n"
       "                      constant uint& v_stride [[buffer(6)]],\n"
       "                      uint2 gid [[thread_position_in_grid]]) {\n"
       "  if (gid.y >= nrow || gid.x >= ncol) return;\n"
       "  S[gid.y * s_stride + gid.x] -= v[gid.y * v_stride] * row[gid.x];\n"
       "}\n";
    NSError* err = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&err];
    if (library == nil || err != nil) {
      NSString* msg = err == nil ? @"unknown Metal compile error" : [err localizedDescription];
      error_message = std::string([msg UTF8String]);
      return;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"rank1_sub"];
    if (function == nil) {
      error_message = "rank1_sub function not found";
      return;
    }
    pipeline = [device newComputePipelineStateWithFunction:function error:&err];
    if (pipeline == nil || err != nil) {
      NSString* msg = err == nil ? @"unknown Metal pipeline error" : [err localizedDescription];
      error_message = std::string([msg UTF8String]);
    }
  });
  if (pipeline == nil) {
    throw std::runtime_error("metal_simpls_resident: failed to build rank1 kernel: " + error_message);
  }
  return pipeline;
}

void metal_rank1_sub(id<MTLDevice> device,
                     id<MTLCommandQueue> queue,
                     MetalMatrix& S,
                     const MetalMatrix& v,
                     const MetalMatrix& row) {
  id<MTLComputePipelineState> pipeline = rank1_sub_pipeline(device);
  uint32_t nrow = static_cast<uint32_t>(S.rows);
  uint32_t ncol = static_cast<uint32_t>(S.cols);
  uint32_t s_stride = static_cast<uint32_t>(S.row_bytes / sizeof(float));
  uint32_t v_stride = static_cast<uint32_t>(v.row_bytes / sizeof(float));

  id<MTLBuffer> nrow_buffer = [device newBufferWithBytes:&nrow length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  id<MTLBuffer> ncol_buffer = [device newBufferWithBytes:&ncol length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  id<MTLBuffer> s_stride_buffer = [device newBufferWithBytes:&s_stride length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  id<MTLBuffer> v_stride_buffer = [device newBufferWithBytes:&v_stride length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:S.buffer offset:0 atIndex:0];
  [encoder setBuffer:v.buffer offset:0 atIndex:1];
  [encoder setBuffer:row.buffer offset:0 atIndex:2];
  [encoder setBuffer:nrow_buffer offset:0 atIndex:3];
  [encoder setBuffer:ncol_buffer offset:0 atIndex:4];
  [encoder setBuffer:s_stride_buffer offset:0 atIndex:5];
  [encoder setBuffer:v_stride_buffer offset:0 atIndex:6];

  MTLSize grid = MTLSizeMake(S.cols, S.rows, 1);
  NSUInteger w = std::min<NSUInteger>(16, pipeline.threadExecutionWidth);
  MTLSize threads = MTLSizeMake(w, 16, 1);
  [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  NSError* err = [command_buffer error];
  if (err != nil) {
    NSString* msg = [err localizedDescription];
    throw std::runtime_error(std::string("metal_simpls_resident: rank1 kernel failed: ") +
                             std::string([msg UTF8String]));
  }
}

} // namespace

bool has_metal_backend() {
  @autoreleasepool {
    id<MTLDevice> device = default_device();
    return device != nil;
  }
}

arma::mat metal_matrix_multiply(const arma::mat& A,
                                const arma::mat& B,
                                const bool transpose_left,
                                const bool transpose_right) {
  const arma::uword a_rows = transpose_left ? A.n_cols : A.n_rows;
  const arma::uword a_cols = transpose_left ? A.n_rows : A.n_cols;
  const arma::uword b_rows = transpose_right ? B.n_cols : B.n_rows;
  const arma::uword b_cols = transpose_right ? B.n_rows : B.n_cols;

  if (a_cols != b_rows) {
    throw std::runtime_error("metal_matrix_multiply: non-conformable matrices");
  }

  @autoreleasepool {
    id<MTLDevice> device = default_device();
    if (device == nil) {
      throw std::runtime_error("metal_matrix_multiply: no Metal device available");
    }

    id<MTLCommandQueue> queue = default_command_queue();
    if (queue == nil) {
      throw std::runtime_error("metal_matrix_multiply: failed to create Metal command queue");
    }

    const NSUInteger m = static_cast<NSUInteger>(a_rows);
    const NSUInteger k = static_cast<NSUInteger>(a_cols);
    const NSUInteger n = static_cast<NSUInteger>(b_cols);

    const NSUInteger a_row_bytes = [MPSMatrixDescriptor rowBytesFromColumns:static_cast<NSUInteger>(A.n_cols) dataType:MPSDataTypeFloat32];
    const NSUInteger b_row_bytes = [MPSMatrixDescriptor rowBytesFromColumns:static_cast<NSUInteger>(B.n_cols) dataType:MPSDataTypeFloat32];
    const NSUInteger c_row_bytes = [MPSMatrixDescriptor rowBytesFromColumns:n dataType:MPSDataTypeFloat32];

    id<MTLBuffer> a_buffer = [device newBufferWithLength:(a_row_bytes * static_cast<NSUInteger>(A.n_rows)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_buffer = [device newBufferWithLength:(b_row_bytes * static_cast<NSUInteger>(B.n_rows)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_buffer = [device newBufferWithLength:(c_row_bytes * m) options:MTLResourceStorageModeShared];

    if (a_buffer == nil || b_buffer == nil || c_buffer == nil) {
      throw std::runtime_error("metal_matrix_multiply: failed to allocate Metal buffers");
    }

    copy_arma_to_mps_buffer(A, a_buffer, a_row_bytes);
    copy_arma_to_mps_buffer(B, b_buffer, b_row_bytes);
    std::fill_n(static_cast<char*>([c_buffer contents]), static_cast<std::size_t>(c_row_bytes * m), 0);

    MPSMatrixDescriptor* a_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(A.n_rows) columns:static_cast<NSUInteger>(A.n_cols) rowBytes:a_row_bytes dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* b_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(B.n_rows) columns:static_cast<NSUInteger>(B.n_cols) rowBytes:b_row_bytes dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* c_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:c_row_bytes dataType:MPSDataTypeFloat32];

    MPSMatrix* a_matrix = [[MPSMatrix alloc] initWithBuffer:a_buffer descriptor:a_desc];
    MPSMatrix* b_matrix = [[MPSMatrix alloc] initWithBuffer:b_buffer descriptor:b_desc];
    MPSMatrix* c_matrix = [[MPSMatrix alloc] initWithBuffer:c_buffer descriptor:c_desc];

    MPSMatrixMultiplication* mult = [[MPSMatrixMultiplication alloc]
      initWithDevice:device
      transposeLeft:transpose_left
      transposeRight:transpose_right
      resultRows:m
      resultColumns:n
      interiorColumns:k
      alpha:1.0
      beta:0.0];

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (command_buffer == nil) {
      throw std::runtime_error("metal_matrix_multiply: failed to create Metal command buffer");
    }

    [mult encodeToCommandBuffer:command_buffer leftMatrix:a_matrix rightMatrix:b_matrix resultMatrix:c_matrix];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    NSError* err = [command_buffer error];
    if (err != nil) {
      NSString* msg = [err localizedDescription];
      throw std::runtime_error(std::string("metal_matrix_multiply: Metal command failed: ") +
                               std::string([msg UTF8String]));
    }

    return copy_mps_buffer_to_arma(c_buffer, a_rows, b_cols, c_row_bytes);
  }
}

arma::mat metal_matrix_multiply(const arma::mat& A, const arma::mat& B) {
  return metal_matrix_multiply(A, B, false, false);
}

arma::mat metal_crossprod(const arma::mat& A, const arma::mat& B) {
  return metal_matrix_multiply(A, B, true, false);
}

Rcpp::List metal_simpls_resident(const arma::mat& X,
                                 const arma::mat& Y,
                                 int ncomp,
                                 int power_iters,
                                 int seed) {
  if (X.n_rows != Y.n_rows) {
    throw std::runtime_error("metal_simpls_resident: X and Y must have the same number of rows");
  }
  const int max_comp = std::max(1, std::min<int>(ncomp, std::min<int>(X.n_cols, X.n_rows - 1)));
  const int q = static_cast<int>(Y.n_cols);
  if (q < 1) {
    throw std::runtime_error("metal_simpls_resident: Y must have at least one column");
  }
  const int iters = std::max(1, power_iters);
  (void) iters;

  @autoreleasepool {
    id<MTLDevice> device = default_device();
    if (device == nil) {
      throw std::runtime_error("metal_simpls_resident: no Metal device available");
    }
    id<MTLCommandQueue> queue = default_command_queue();
    if (queue == nil) {
      throw std::runtime_error("metal_simpls_resident: failed to create Metal command queue");
    }

    MetalMatrix Xg = make_matrix_from_arma(device, X);
    MetalMatrix Yg = make_matrix_from_arma(device, Y);
    MetalMatrix Sg = metal_matmul(device, queue, Xg, Yg, true, false);

    arma::mat RR(X.n_cols, max_comp, arma::fill::zeros);
    arma::mat QQ(Y.n_cols, max_comp, arma::fill::zeros);
    arma::mat VV(X.n_cols, max_comp, arma::fill::zeros);
    arma::cube B(X.n_cols, Y.n_cols, max_comp, arma::fill::zeros);
    arma::mat Bcur(X.n_cols, Y.n_cols, arma::fill::zeros);

    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::normal_distribution<float> normal(0.0f, 1.0f);

    MetalMatrix vbuf = make_matrix(device, static_cast<NSUInteger>(q), 1);
    MetalMatrix rbuf = make_matrix(device, static_cast<NSUInteger>(X.n_cols), 1);
    MetalMatrix ttbuf = make_matrix(device, static_cast<NSUInteger>(X.n_rows), 1);
    MetalMatrix vvbuf = make_matrix(device, static_cast<NSUInteger>(X.n_cols), 1);
    bool has_rr_prev = false;
    arma::vec rr_prev;

    for (int a = 0; a < max_comp; ++a) {
      arma::vec rr(X.n_cols);
      if (has_rr_prev && rr_prev.n_elem == X.n_cols) {
        rr = rr_prev;
      } else {
        for (arma::uword j = 0; j < X.n_cols; ++j) {
          rr[j] = static_cast<double>(normal(rng));
        }
      }
      double rnorm = arma::norm(rr, 2);
      if (!std::isfinite(rnorm) || rnorm <= std::numeric_limits<double>::epsilon()) break;
      rr /= rnorm;
      write_metal_col_vector(rbuf, rr);
      for (int it = 0; it < iters; ++it) {
        MetalMatrix zbuf = metal_matmul(device, queue, Sg, rbuf, true, false);
        rbuf = metal_matmul(device, queue, Sg, zbuf, false, false);
        rr = copy_metal_vector(rbuf);
        rnorm = arma::norm(rr, 2);
        if (!std::isfinite(rnorm) || rnorm <= std::numeric_limits<double>::epsilon()) break;
        rr /= rnorm;
        write_metal_col_vector(rbuf, rr);
      }
      rr = copy_metal_vector(rbuf);
      rnorm = arma::norm(rr, 2);
      if (!std::isfinite(rnorm) || rnorm <= std::numeric_limits<double>::epsilon()) break;
      rr /= rnorm;

      write_metal_col_vector(rbuf, rr);

      ttbuf = metal_matmul(device, queue, Xg, rbuf, false, false);
      arma::vec tt = copy_metal_vector(ttbuf);
      double tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= std::numeric_limits<double>::epsilon()) break;
      tt /= tnorm;
      rr /= tnorm;
      rr_prev = rr;
      has_rr_prev = true;
      write_metal_col_vector(ttbuf, tt);

      MetalMatrix ppbuf = metal_matmul(device, queue, Xg, ttbuf, true, false);
      MetalMatrix qqbuf = metal_matmul(device, queue, Yg, ttbuf, true, false);
      arma::vec pp = copy_metal_vector(ppbuf);
      arma::vec qq = copy_metal_vector(qqbuf);

      arma::vec vv = pp;
      if (a > 0) {
        arma::mat Vprev = VV.cols(0, a - 1);
        vv -= Vprev * (Vprev.t() * pp);
      }
      double vvnorm = arma::norm(vv, 2);
      if (!std::isfinite(vvnorm) || vvnorm <= std::numeric_limits<double>::epsilon()) break;
      vv /= vvnorm;
      write_metal_col_vector(vvbuf, vv);

      MetalMatrix rowbuf = metal_matmul(device, queue, vvbuf, Sg, true, false);
      metal_rank1_sub(device, queue, Sg, vvbuf, rowbuf);

      RR.col(a) = rr;
      QQ.col(a) = qq;
      VV.col(a) = vv;
      Bcur += rr * qq.t();
      B.slice(a) = Bcur;
    }

    return Rcpp::List::create(
      Rcpp::Named("R") = RR,
      Rcpp::Named("Q") = QQ,
      Rcpp::Named("V") = VV,
      Rcpp::Named("B") = B
    );
  }
}

} // namespace fastpls_svd
