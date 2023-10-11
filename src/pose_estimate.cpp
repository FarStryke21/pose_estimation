#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>

int main(int argc, char** argv)
{
    // Load the source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPCDFile("part.pcd", *source_cloud);
    pcl::io::loadPCDFile("part_scan.pcd", *target_cloud);

    // Initialize the ICP object
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);

    // Set the maximum number of iterations (adjust as needed)
    icp.setMaximumIterations(50);

    // Set the transformation epsilon (adjust as needed)
    icp.setTransformationEpsilon(1e-8);

    // Align the clouds
    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    icp.align(aligned_cloud);

    if (icp.hasConverged())
    {
        std::cout << "ICP converged with a score of " << icp.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:\n" << icp.getFinalTransformation() << std::endl;

        // Visualize the aligned cloud
        pcl::visualization::CloudViewer viewer("ICP Result");
        viewer.showCloud(aligned_cloud.makeShared());
        while (!viewer.wasStopped())
        {
        }
    }
    else
    {
        std::cerr << "ICP did not converge" << std::endl;
    }

    return 0;
}
