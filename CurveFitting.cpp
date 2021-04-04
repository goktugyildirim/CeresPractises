#include <iostream>
#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <vector>


// Generate data from the function: f(x) = 8*x^2 + 3*x + 5 -> solution a:8 b:3 c:5
struct XGenerator {
    double x = 0;
    void operator() (double& x_)
    { x = x + 0.01;
      x_ = x;
      //std::cout << "x: " << x_ << std::endl;
    }
};

struct Function {
    double operator () (double x)
    {
        double y = 8*pow(x,2) + 3*x + 5;
        //std::cout << "y: " << y << std::endl;
        return y;
    }
};

XGenerator increment;
Function f;

struct PolynomialResidual {

    PolynomialResidual(double x_, double y_)
    : x_real(x_), y_real(y_) {}

    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {

        residual[0] = y_real - (a[0]*pow(x_real,2) + b[0]*x_real + c[0]);
        return true;
    }

private:
    const double x_real;
    const double y_real;

};



int main() {
    size_t sample_count = 100;
    std::vector<double> x_values(sample_count);
    std::vector<double> y_values(sample_count);

    std::for_each(x_values.begin(),x_values.end(),increment);
    std::transform(x_values.begin(),x_values.end(),y_values.begin(),f);

    double a = 0;
    double b = 0;
    double c = 0;

    ceres::Problem problem;
    for (int i = 0; i < sample_count; ++i) {
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PolynomialResidual, 1, 1, 1, 1>(new PolynomialResidual(x_values[i], y_values[i])),
                new ceres::CauchyLoss(0.5),
                &a, &b, &c
                );
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.num_threads = 12;
    options.linear_solver_type= ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial a: " << 0.0 << " b: " << 0.0 << " c: " << 0.0 << "\n";
    std::cout << "Final   a: " << a << " b: " << b << " c: " << c << "\n";


    return 0;
}
