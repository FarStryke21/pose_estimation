#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

int main(int argc, char** argv)
{
    // Load the source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPLYFile("owl_test.ply", *source_cloud);        // Base File
    pcl::io::loadPLYFile("owl_scan_1.ply", *target_cloud);   // Scan File
    std::cout << "Loaded " << std::endl;

    // Magnify pointcloud by 1000 times
    float magnification_factor = 1000.0;
    for (size_t i = 0; i < target_cloud->points.size(); ++i) {
        target_cloud->points[i].x *= magnification_factor;
        target_cloud->points[i].y *= magnification_factor;
        target_cloud->points[i].z *= magnification_factor;
    }

    // ----------------------Downsample the target cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(target_cloud); 
    voxel_grid.setLeafSize(5, 5, 5);
    voxel_grid.filter(*downsampled_target_cloud);
    *target_cloud = *downsampled_target_cloud;

    // // -------------------Shift the target cloud 10 units along the x axis using an eigen transformation
    // Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
    // transform_1(0, 3) = 100.0;
    // //Print the Transformation Matrix
    // std::cout << "Transformation Matrix:\n" << transform_1 << std::endl;
    // // perform the shift
    // pcl::transformPointCloud(*target_cloud, *target_cloud, transform_1);

    // ---------------------Initialize the ICP object
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);

    // Set the maximum number of iterations (adjust as needed)
    icp.setMaximumIterations(10);

    // Set the transformation epsilon (adjust as needed)
    icp.setTransformationEpsilon(1e-5);

    // Align the clouds
    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    std::cout << "ICP Starting ... " << std::endl;
    icp.align(aligned_cloud);

    if (icp.hasConverged())
    {
        std::cout << "ICP converged with a score of " << icp.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:\n" << icp.getFinalTransformation() << std::endl;

        // Visualize the aligned cloud
        pcl::visualization::CloudViewer viewer("ICP Result");
        // Show thaligned cloud in a particular color
        // viewer.showCloud(source_cloud, "source");
        // viewer.showCloud(target_cloud, "result");
        viewer.showCloud(aligned_cloud.makeShared(), "Aligned");

        // Use the transformation matrix to transform the target cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*target_cloud, *transformed_cloud, icp.getFinalTransformation());
        viewer.showCloud(transformed_cloud, "Transformed");

        while (!viewer.wasStopped())
        {
        }
    }
    else
    {
        std::cerr << "ICP did not converge" << std::endl;
    }

    // Plain Visualiser
    // pcl::visualization::CloudViewer viewer("ICP Result");
    // viewer.showCloud(source_cloud, "source");
    // viewer.showCloud(target_cloud, "result");  
    // while (!viewer.wasStopped())
    // {
    // }


    return 0;
}
