#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>

int main(int argc, char** argv)
{
    // Load the source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPLYFile("owl_test.ply", *source_cloud);
    pcl::io::loadPLYFile("owl_scan_1.ply", *target_cloud);

    // // Initialize the ICP object
    // pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    // icp.setInputSource(source_cloud);
    // icp.setInputTarget(target_cloud);

    // // Set the maximum number of iterations (adjust as needed)
    // icp.setMaximumIterations(5);

    // // Set the transformation epsilon (adjust as needed)
    // icp.setTransformationEpsilon(1e-5);

    // // Align the clouds
    // pcl::PointCloud<pcl::PointXYZ> aligned_cloud;

    // for (int i = 0; i < icp.getMaximumIterations(); ++i)
    // {
    //     icp.align(aligned_cloud);

    //     std::cout << "Iteration " << i+1 << ": ";
    //     if (icp.hasConverged())
    //     {
    //         std::cout << "Converged with score " << icp.getFitnessScore() << std::endl;
    //     }
    //     else
    //     {
    //         std::cout << "Not converged" << std::endl;
    //     }
        
    //     Eigen::Matrix4f transformation_matrix = icp.getFinalTransformation();
    //     std::cout << "Transformation matrix:\n" << transformation_matrix << std::endl;
    // }

    // if (icp.hasConverged())
    // {
    //     std::cout << "ICP converged with a score of " << icp.getFitnessScore() << std::endl;
    //     std::cout << "Transformation matrix:\n" << icp.getFinalTransformation() << std::endl;

    //     // Visualize the aligned cloud
    //     pcl::visualization::CloudViewer viewer("ICP Result");
    //     viewer.showCloud(aligned_cloud.makeShared());
    //     // viewer.showCloud(target_cloud);
    //     while (!viewer.wasStopped())
    //     {
    //     }
    // }
    // else
    // {
    //     std::cerr << "ICP did not converge" << std::endl;
    // }
    pcl::visualization::CloudViewer viewer("My Viewer");
    viewer.showCloud(source_cloud);
    while (!viewer.wasStopped())
    {
    }
    return 0;
}
