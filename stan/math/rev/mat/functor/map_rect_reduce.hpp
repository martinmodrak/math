#pragma once

namespace stan {
  namespace math {
    template <typename F>
    struct map_rect_reduce<F, var, var> {
      static std::size_t get_output_size(std::size_t num_shared_params, std::size_t num_job_specific_params) {
        return(1+num_shared_params+num_job_specific_params);
      }
      static matrix_d apply(const vector_d& shared_params, const vector_d& job_specific_params, const std::vector<double>& x_r, const std::vector<int>& x_i) {
        const size_type num_shared_params = shared_params.rows();
        const size_type num_job_specific_params = job_specific_params.rows();
        const size_type num_params = num_shared_params  + num_job_specific_params;
        matrix_d out(1+num_params,0);

        try {
          start_nested();
          vector_v shared_params_v(shared_params);
          vector_v job_specific_params_v(job_specific_params);

          std::vector<var> z_vars(num_params);
          std::vector<double> z_grad(num_params);

          for(size_type i = 0; i < num_shared_params; ++i)
            z_vars[i] = shared_params_v(i);
          for(size_type i = 0; i < num_job_specific_params; ++i)
            z_vars[num_shared_params + i] = job_specific_params_v(i);

          vector_v fx_v = F::apply(shared_params_v, job_specific_params_v, x_r, x_i);

          const size_t size_f = fx_v.rows();

          out.resize(Eigen::NoChange, size_f);

          for(size_type i = 0; i < size_f; ++i) {
            set_zero_all_adjoints_nested();
            fx_v(i).grad(z_vars, z_grad);
            out(0,i) = fx_v(i).val();
            out.block(1,i,num_shared_params,1) = Eigen::Map<vector_d>(z_grad.data(), num_shared_params);
            out.block(1+num_shared_params,i,num_job_specific_params,1) = Eigen::Map<vector_d>(z_grad.data() + num_shared_params, num_job_specific_params);
          }
          recover_memory_nested();
        } catch(const std::exception& e) {
          recover_memory_nested();
          throw;
        }
        return( out );
      }
    };
  }
}
