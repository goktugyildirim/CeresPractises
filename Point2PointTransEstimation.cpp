//
// Created by goktug on 16.03.2021.
//

#include "Point2PointTransEstimation.h"




namespace OptimizationHelper
{
    template<typename T>
    CorrespondencePointTransformationEstimation<T>::CorrespondencePointTransformationEstimation()
                        :cloud_source(new Cloud),
                         cloud_target(new Cloud)
    {

    }

    template<typename T>
    void CorrespondencePointTransformationEstimation<T>::setSourceCorrespondence(const CloudPtr& cloud)
    {
        vector_points_source.reserve(cloud->size());

        for(const T pcl_point : cloud->points)
        {
            Eigen::Matrix<double,4,1> vector_point;
            vector_point(0) = pcl_point.x;
            vector_point(1) = pcl_point.y;
            vector_point(2) = pcl_point.z;
            vector_point(3) = 1;
            vector_points_source.push_back(vector_point);
        }

    }

    template<typename T>
    void CorrespondencePointTransformationEstimation<T>::setTargetCorrespondence(
            const CorrespondencePointTransformationEstimation::CloudPtr &cloud)
    {
        vector_points_target.reserve(cloud->size());

        for(const T pcl_point : cloud->points)
        {
            Eigen::Matrix<double,4,1> vector_point;
            vector_point(0) = pcl_point.x;
            vector_point(1) = pcl_point.y;
            vector_point(2) = pcl_point.z;
            vector_point(3) = 1;
            vector_points_target.push_back(vector_point);
        }



    }

    template<typename T> void
    CorrespondencePointTransformationEstimation<T>::estimateTransformation()
    {
        double params_q[4] = {0.8,0.8,1,1};
        double params_translation[3] = {0.2,0.2,0.1};

        size_t count_point = vector_points_source.size();

        ceres::Problem problem;

        for (int i = 0; i < count_point; ++i) {
            problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<MyCostFunctor, 3,4,3>
                    (
                    new MyCostFunctor(vector_points_source[i],vector_points_target[i])),
                    nullptr,
                    params_q,
                    params_translation);
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 24;
        options.num_threads = 12;
        options.linear_solver_type= ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        std::cout << "Solution of translation: " << std::endl;
        std::cout << params_translation[0] << " " <<
        params_translation[1] << " " << params_translation[2] << std::endl;

        std::cout << "Solution of quaternion(x y z w) order: " << std::endl;
        std::cout <<
        params_q[0] << " " <<
        params_q[1] << " " <<
        params_q[2] << " " <<
        params_q[3] << " " << std::endl;





    }


}


int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);

    for(int i=0; i<10000; i++)
    {
        pcl::PointXYZ p;
        p.x = i;
        p.y = i+1;
        p.z = i+2;
        cloud_source->points.push_back(p);
    }

    Eigen::Matrix4d mat_static;
    mat_static.setIdentity();
    Eigen::Quaternion<double> q;
    // verdi??in initial de??erler bunlara g??re olmal??!
    q.x() = 0.8;
    q.y() = 0.8;
    q.z() = 1;
    q.w() = 1;
    q = q.normalized();
    std::cout << "Target quaternion(x y z w): " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    Eigen::Matrix3d rotationMatrix = q.matrix();
    mat_static.topLeftCorner(3,3) = rotationMatrix;
    mat_static(0,3) = 0.25;
    mat_static(1,3) = 0.25;
    mat_static(2,3) = 0.25;
    std::cout << "Target translation: " << mat_static(0,3) << " " << mat_static(1,3) << " " << mat_static(2,3) << std::endl;

    pcl::transformPointCloud(*cloud_source,*cloud_target,mat_static);
    OptimizationHelper::CorrespondencePointTransformationEstimation<pcl::PointXYZ> estimator;

    estimator.setSourceCorrespondence(cloud_source);
    estimator.setTargetCorrespondence(cloud_target);
    estimator.estimateTransformation();

    std::cout << "Optimization is done!" << std::endl;


    return 0;
}