#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>

// include the header file for the utils
#include "pose_estimation/estimate_utils.h"

int main(int argc, char** argv)
{
    // Load the source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Use the load point cloud function from utils.h
    target_cloud = loadPointCloud("data/owl_sampled.ply");
    source_cloud = loadPointCloud("data/owl_scan_1.ply");
    
    scalePointCloud(source_cloud, 1000.0);
    downsamplePointCloud(source_cloud);
    removeOutliers(source_cloud);

    // Make an independent copy of the source cloud which is not linked to the source cloud anymore
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);
    *source_cloud_copy = *source_cloud;

    source_cloud = change_coordinate_frame(source_cloud);


    // Call the Pose Estimate function which uses SAC IA
    Eigen::Matrix4f transform_1 = computeTransformation_SACIA(source_cloud, target_cloud);  
    Eigen::Matrix4f transform_2 = computeTransformation_ICP(source_cloud, target_cloud);

    std::cout << "\nBeginning Processing of Transforms... \n" << std::endl;
    // Transform 1 given mapping from initial to mid and Tranbsform 2 given mapping from mid to final. Find the mapping from initial to final
    Eigen::Matrix4f transform_3 = transform_2 * transform_1;

    // Transform 3 is the complete transformation matrix. Extract the rotation part from it
    Eigen::Matrix3f rotation_matrix = transform_3.block<3, 3>(0, 0);
    Eigen::Quaternionf quaternion(rotation_matrix);

    // Convert the rotation matrix to roll, pitch and yaw in degrees
    float roll = atan2(rotation_matrix(2, 1), rotation_matrix(2, 2)) * 180 / M_PI;
    float pitch = atan2(-rotation_matrix(2, 0), sqrt(rotation_matrix(2, 1) * rotation_matrix(2, 1) + rotation_matrix(2, 2) * rotation_matrix(2, 2))) * 180 / M_PI;
    float yaw = atan2(rotation_matrix(1, 0), rotation_matrix(0, 0)) * 180 / M_PI;
    std::cout << "Roll : " << roll << std::endl;
    std::cout << "Pitch : " << pitch << std::endl;
    std::cout << "Yaw : " << yaw << std::endl;

    // Visualize the aligned cloud
    pcl::visualization::PCLVisualizer viewer("SAC IA Result");
    viewer.addPointCloud(target_cloud, "Target Cloud");
    viewer.addPointCloud(source_cloud, "Final Source Cloud");
    viewer.addPointCloud(source_cloud_copy, "Initial Source Cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Target Cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Final Source Cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "Initial Source Cloud");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addCoordinateSystem(25.0);

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }

    std::cout << "\nPose Error in Yaw : "<< yaw << std::endl;
    return 0;
}
