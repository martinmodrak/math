#ifndef STAN_MATH_OPENCL_MATRIX_CL_HPP
#define STAN_MATH_OPENCL_MATRIX_CL_HPP
#ifdef STAN_OPENCL
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/opencl/is_matrix_cl.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/prim/arr/fun/vec_concat.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <cl.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

/** \ingroup opencl
 *  @file stan/math/opencl/matrix_cl.hpp
 *  @brief The matrix_cl class - allocates memory space on the OpenCL device,
 *    functions for transfering matrices to and from OpenCL devices
 */
namespace stan {
namespace math {

/** \ingroup opencl
 * Represents a matrix on the OpenCL device.
 * @tparam T an arithmetic type for the type stored in the OpenCL buffer.
 */
template <typename T>
class matrix_cl<T, require_arithmetic_t<T>> {
 private:
  cl::Buffer buffer_cl_;  // Holds the allocated memory on the device
  int rows_{0};
  int cols_{0};
  // Holds info on if matrix is a special type
  matrix_cl_view view_{matrix_cl_view::Entire};
  mutable std::vector<cl::Event> write_events_;  // Tracks write jobs
  mutable std::vector<cl::Event> read_events_;   // Tracks reads

 public:
  using Scalar = T;
  using type = T;
  // Forward declare the methods that work in place on the matrix
  template <matrix_cl_view matrix_view = matrix_cl_view::Entire>
  inline void zeros();
  template <matrix_cl_view matrix_view = matrix_cl_view::Entire>
  inline void zeros_strict_tri();
  template <TriangularMapCL triangular_map = TriangularMapCL::LowerToUpper>
  inline void triangular_transpose();

  inline void sub_block(const matrix_cl<T, require_arithmetic_t<T>>& A,
                        size_t A_i, size_t A_j, size_t this_i, size_t this_j,
                        size_t nrows, size_t ncols);
  int rows() const { return rows_; }

  int cols() const { return cols_; }

  int size() const { return rows_ * cols_; }

  const matrix_cl_view& view() const { return view_; }

  void view(const matrix_cl_view& view) { view_ = view; }

  /** \ingroup opencl
   * Clear the write events from the event stacks.
   */
  inline void clear_write_events() const {
    write_events_.clear();
    return;
  }

  /** \ingroup opencl
   * Clear the read events from the event stacks.
   */
  inline void clear_read_events() const {
    read_events_.clear();
    return;
  }

  /** \ingroup opencl
   * Clear the write events from the event stacks.
   */
  inline void clear_read_write_events() const {
    read_events_.clear();
    write_events_.clear();
    return;
  }

  /** \ingroup opencl
   * Get the events from the event stacks.
   * @return The write event stack.
   */
  inline const std::vector<cl::Event>& write_events() const {
    return write_events_;
  }

  /** \ingroup opencl
   * Get the events from the event stacks.
   * @return The read/write event stack.
   */
  inline const std::vector<cl::Event>& read_events() const {
    return read_events_;
  }

  /** \ingroup opencl
   * Get the events from the event stacks.
   * @return The read/write event stack.
   */
  inline const std::vector<cl::Event> read_write_events() const {
    return vec_concat(this->read_events(), this->write_events());
  }

  /** \ingroup opencl
   * Add an event to the read event stack.
   * @param new_event The event to be pushed on the event stack.
   */
  inline void add_read_event(cl::Event new_event) const {
    this->read_events_.push_back(new_event);
  }

  /** \ingroup opencl
   * Add an event to the write event stack.
   * @param new_event The event to be pushed on the event stack.
   */
  inline void add_write_event(cl::Event new_event) const {
    this->write_events_.push_back(new_event);
  }

  /** \ingroup opencl
   * Add an event to the read/write event stack.
   * @param new_event The event to be pushed on the event stack.
   */
  inline void add_read_write_event(cl::Event new_event) const {
    this->read_events_.push_back(new_event);
    this->write_events_.push_back(new_event);
  }

  /** \ingroup opencl
   * Waits for the write events and clears the read event stack.
   */
  inline void wait_for_write_events() const {
    cl::CommandQueue queue = opencl_context.queue();
    cl::Event copy_event;
    queue.enqueueBarrierWithWaitList(&this->write_events(), &copy_event);
    copy_event.wait();
    write_events_.clear();
    return;
  }

  /** \ingroup opencl
   * Waits for the read events and clears the read event stack.
   */
  inline void wait_for_read_events() const {
    cl::CommandQueue queue = opencl_context.queue();
    cl::Event copy_event;
    queue.enqueueBarrierWithWaitList(&this->read_events(), &copy_event);
    copy_event.wait();
    read_events_.clear();
    return;
  }

  /** \ingroup opencl
   * Waits for read and write events to finish and clears the read, write, and
   * read/write event stacks.
   */
  inline void wait_for_read_write_events() const {
    cl::CommandQueue queue = opencl_context.queue();
    cl::Event copy_event;
    const std::vector<cl::Event> mat_events = this->read_write_events();
    queue.enqueueBarrierWithWaitList(&mat_events, &copy_event);
    copy_event.wait();
    read_events_.clear();
    write_events_.clear();
    return;
  }

  const cl::Buffer& buffer() const { return buffer_cl_; }
  cl::Buffer& buffer() { return buffer_cl_; }

  matrix_cl() {}
  /** \ingroup opencl
   * Construct a matrix_cl<T> from an existing cl::Buffer object. The matrix
   * directly uses given buffer - no copying is done.
   *
   * @param A the cl::Buffer object to construct the matrix from
   * @param R number of rows
   * @param C number of columns
   * @param partial_view view of the matrix
   */
  matrix_cl(cl::Buffer& A, const int R, const int C,
            matrix_cl_view partial_view = matrix_cl_view::Entire)
      : buffer_cl_(A), rows_(R), cols_(C), view_(partial_view) {}

  /**
   * Copy constructor.
   * @param A matrix_cl to copy
   */
  matrix_cl(const matrix_cl<T>& A)
      : rows_(A.rows()), cols_(A.cols()), view_(A.view()) {
    if (A.size() == 0) {
      return;
    }
    initialize_buffer(A);
  }

  /**
   * Move constructor.
   * @param A matrix_cl to move
   */
  matrix_cl(matrix_cl<T>&& A)
      : buffer_cl_(std::move(A.buffer_cl_)),
        rows_(A.rows_),
        cols_(A.cols_),
        view_(A.view_),
        write_events_(std::move(A.write_events_)),
        read_events_(std::move(A.read_events_)) {}

  /** \ingroup opencl
   * Constructor for the matrix_cl that creates a copy of a std::vector of Eigen
   * matrices on the OpenCL device. Each matrix is flattened into one column
   * of the resulting matrix_cl.
   *
   * @param A the vector of Eigen matrices
   *
   * @throw <code>std::invalid_argument</code> if the
   * matrices do not have matching dimensions
   * @throw <code>std::system_error</code> if the memory on the device could not
   * be allocated
   */
  template <typename Vec, require_std_vector_vt<is_eigen, Vec>...,
            require_same_st<Vec, T>...>
  explicit matrix_cl(Vec&& A) try : rows_(A.empty() ? 0 : A[0].size()),
                                    cols_(A.size()) {
    if (this->size() == 0) {
      return;
    }
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    buffer_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * size());
    for (int i = 0, offset_size = 0; i < cols_; i++, offset_size += rows_) {
      check_size_match("matrix constructor", "input rows", A[i].size(),
                       "matrix_cl rows", rows_);
      cl::Event write_event;
      queue.enqueueWriteBuffer(
          buffer_cl_,
          opencl_context.in_order() || std::is_rvalue_reference<Vec&&>::value,
          sizeof(T) * offset_size, sizeof(T) * rows_, A[i].data(), nullptr,
          &write_event);
      this->add_write_event(write_event);
    }
  } catch (const cl::Error& e) {
    check_opencl_error("matrix constructor", e);
  }

  /** \ingroup opencl
   * Constructor for the matrix_cl that
   * only allocates the buffer on the OpenCL device.
   * Regardless of `partial_view`, whole matrix is stored.
   *
   * @param rows number of matrix rows, must be greater or equal to 0
   * @param cols number of matrix columns, must be greater or equal to 0
   * @param partial_view which part of the matrix is used
   *
   * @throw <code>std::system_error</code> if the memory on the device could not
   * be allocated
   *
   */
  matrix_cl(const int rows, const int cols,
            matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(rows), cols_(cols), view_(partial_view) {
    if (size() == 0) {
      return;
    }
    cl::Context& ctx = opencl_context.context();
    try {
      buffer_cl_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * rows_ * cols_);
    } catch (const cl::Error& e) {
      check_opencl_error("matrix constructor", e);
    }
  }

  /** \ingroup opencl
   * Constructor for the matrix_cl that
   * creates a copy of the Eigen matrix on the OpenCL device.
   * Regardless of `partial_view`, whole matrix is stored.
   *
   * @tparam T type of data in the \c Eigen \c Matrix
   * @param A the \c Eigen \c Matrix
   * @param partial_view which part of the matrix is used
   *
   * @throw <code>std::system_error</code> if the memory on the device could not
   * be allocated
   */
  template <typename Mat, require_eigen_t<Mat>..., require_same_vt<Mat, T>...>
  explicit matrix_cl(Mat&& A,
                     matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(A.rows()), cols_(A.cols()), view_(partial_view) {
    initialize_buffer<std::is_rvalue_reference<Mat&&>::value
                      || !is_eigen_matrix<Mat>::value>(A.eval().data());
  }

  /** \ingroup opencl
   * Constructor for the matrix_cl that
   * creates a copy of a scalar on the OpenCL device.
   * Regardless of `partial_view`, whole matrix is stored.
   *
   * @param A the scalar
   * @param partial_view which part of the matrix is used
   *
   * @throw <code>std::system_error</code> if the memory on the device could not
   * be allocated
   */
  template <typename Scal,
            typename = require_same_t<T, std::remove_reference_t<Scal>>>
  explicit matrix_cl(Scal&& A,
                     matrix_cl_view partial_view = matrix_cl_view::Diagonal)
      : rows_(1), cols_(1), view_(partial_view) {
    initialize_buffer<std::is_rvalue_reference<Scal&&>::value>(&A);
  }

  /** \ingroup opencl
   * Construct a matrix_cl of size Nx1 from \c std::vector
   *
   * @param A Standard vector
   * @param partial_view which part of the matrix is used
   *
   * @throw <code>std::system_error</code> if the memory on the device could not
   * be allocated
   */
  template <typename Vec, require_std_vector_t<Vec>...,
            require_same_vt<Vec, T>...>
  explicit matrix_cl(Vec&& A,
                     matrix_cl_view partial_view = matrix_cl_view::Entire)
      : matrix_cl(std::forward<Vec>(A), A.size(), 1) {}

  /**
   * Construct from \c std::vector with given rows and columns
   *
   * @param A Standard vector
   * @param R Number of rows the matrix should have.
   * @param C Number of columns the matrix should have.
   * @param partial_view which part of the matrix is used
   *
   * @throw <code>std::system_error</code> if the memory on the device could not
   * be allocated
   */
  template <typename Vec, require_std_vector_t<Vec>...,
            require_same_vt<Vec, T>...>
  explicit matrix_cl(Vec&& A, const int& R, const int& C,
                     matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(R), cols_(C), view_(partial_view) {
    initialize_buffer<std::is_rvalue_reference<Vec&&>::value>(A.data());
  }

  /** \ingroup opencl
   * Construct from \c array of doubles with given rows and columns
   *
   * @param A array of doubles
   * @param R Number of rows the matrix should have.
   * @param C Number of columns the matrix should have.
   * @param partial_view which part of the matrix is used
   *
   * @throw <code>std::system_error</code> if the memory on the device could not
   * be allocated
   */
  template <typename U, require_same_t<T, U>...>
  explicit matrix_cl(const U* A, const int& R, const int& C,
                     matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(R), cols_(C), view_(partial_view) {
    initialize_buffer(A);
  }

  /** \ingroup opencl
   * Move assignment operator.
   */
  matrix_cl<T>& operator=(matrix_cl<T>&& a) {
    view_ = a.view();
    rows_ = a.rows();
    cols_ = a.cols();
    this->wait_for_read_write_events();
    buffer_cl_ = std::move(a.buffer_cl_);
    write_events_ = std::move(a.write_events_);
    read_events_ = std::move(a.read_events_);
    return *this;
  }

  /** \ingroup opencl
   * Copy assignment operator.
   */
  matrix_cl<T>& operator=(const matrix_cl<T>& a) {
    if (a.size() == 0) {
      return *this;
    }
    this->view_ = a.view();
    this->rows_ = a.rows();
    this->cols_ = a.cols();
    this->wait_for_read_write_events();
    initialize_buffer(a);
    return *this;
  }

 private:
  /**
   * Initializes the OpencL buffer of this matrix by copying the data from given
   * buffer. Assumes that size of \c this is already set and matches the
   * buffer size.
   * @tparam in_order whether copying must be done in order
   * @param A pointer to buffer
   */
  template <bool in_order = false>
  void initialize_buffer(const T* A) {
    if (size() == 0) {
      return;
    }
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    try {
      buffer_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * size());
      cl::Event transfer_event;
      queue.enqueueWriteBuffer(buffer_cl_,
                               opencl_context.in_order() || in_order, 0,
                               sizeof(T) * size(), A, nullptr, &transfer_event);
      this->add_write_event(transfer_event);
    } catch (const cl::Error& e) {
      check_opencl_error("initialize_buffer", e);
    }
  }

  /**
   * Initializes the OpencL buffer of this matrix by copying the data from given
   * matrix_cl. Assumes that size of \c this is already set and matches the
   * size of given matrix.
   * @param A matrix_cl
   */
  void initialize_buffer(const matrix_cl<T>& A) {
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue queue = opencl_context.queue();
    try {
      buffer_cl_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * this->size());
      cl::Event cstr_event;
      queue.enqueueCopyBuffer(A.buffer(), this->buffer(), 0, 0,
                              A.size() * sizeof(T), &A.write_events(),
                              &cstr_event);
      this->add_write_event(cstr_event);
      A.add_read_event(cstr_event);
    } catch (const cl::Error& e) {
      check_opencl_error("copy (OpenCL)->(OpenCL)", e);
    }
  }
};

template <typename T>
using matrix_cl_prim = matrix_cl<T, require_arithmetic_t<T>>;

template <typename T>
using matrix_cl_fp = matrix_cl<T, require_floating_point_t<T>>;

}  // namespace math
}  // namespace stan

#endif
#endif
