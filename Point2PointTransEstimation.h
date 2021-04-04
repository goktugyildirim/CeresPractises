//
// Created by goktug on 16.03.2021.
//

#ifndef CERES_PRACTISE_CORRESPONDENCEPOINTTRANSFORMATIONESTIMATION_H
#define CERES_PRACTISE_CORRESPONDENCEPOINTTRANSFORMATIONESTIMATION_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <ceres/cost_function_to_functor.h>


#include <Eigen/StdVector>

namespace OptimizationHelper
{
    template<typename T>
    class CorrespondencePointTransformationEstimation
    {
    public:
        typedef typename pcl::PointCloud<T> Cloud;
        typedef typename pcl::PointCloud<T>::Ptr CloudPtr;

        explicit CorrespondencePointTransformationEstimation<T>
                ();

        void setSourceCorrespondence(const CloudPtr& cloud);
        void setTargetCorrespondence(const CloudPtr& cloud);
        void estimateTransformation();

    private:
        CloudPtr cloud_source;
        CloudPtr cloud_target;

        std::vector<Eigen::Matrix<double,4,1>, Eigen::aligned_allocator<Eigen::Matrix<double,4,1>>> vector_points_source;
        std::vector<Eigen::Matrix<double,4,1>, Eigen::aligned_allocator<Eigen::Matrix<double,4,1>>> vector_points_target;


        struct MyCostFunctor
        {
        public:
            MyCostFunctor(Eigen::Vector4d point_source_,
                          Eigen::Vector4d point_target_)
                          :point_source(point_source_),
                          point_target(point_target_)
                          {}

            template <typename T_>
            bool operator () (const T_* const params_q,
                              const T_* const params_translation,
                              T_* residual) const
            {
                Eigen::Matrix<T_,4,4> mat_predicted_all = Eigen::Matrix<T_,4,4>::Identity();

                // set rotation
                /* For some historical reasons AngleAxisToRotation is a ColumnMajor matrix and the
                 * QuaternionToRotation is a row major matrix.*/
                T_ mat_rot[9];
                ceres::QuaternionToRotation(params_q,mat_rot);
                Eigen::Map<const Eigen::Matrix<T_, 3, 3, Eigen::RowMajor>> mat_rot_predicted(mat_rot);
                mat_predicted_all.template topLeftCorner<3,3>() = mat_rot_predicted;

                // set translation
                mat_predicted_all(0,3) = params_translation[0];
                mat_predicted_all(1,3) = params_translation[1];
                mat_predicted_all(2,3) = params_translation[2];

                // evaluate
                Eigen::Matrix<T_, 4, 1> point_transformed = mat_predicted_all*point_source.cast<T_>();

                residual[0] = point_transformed(0) - point_target(0);
                residual[1] = point_transformed(1) - point_target(1);
                residual[2] = point_transformed(2) - point_target(2);
                return true;
            }

        private:
            const Eigen::Matrix<double,4,1> point_source;
            const Eigen::Matrix<double,4,1> point_target;
        };

    };

}



#endif //CERES_PRACTISE_CORRESPONDENCEPOINTTRANSFORMATIONESTIMATION_H
