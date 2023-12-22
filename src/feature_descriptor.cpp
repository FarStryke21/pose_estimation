// Write a program that loads a pointcloud and finds the FPFH feautres and then show them in the viewer
// Hint: Use the functions from utils.h to load the pointcloud and then use the functions from feature_descriptor.h to find the FPFH features

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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Use the load point cloud function from utils.h
    cloud = loadPointCloud("data/owl_sampled.ply");
    // scalePointCloud(cloud, 1000.0);
    cout << "Point cloud loaded" << endl;

    // Compute the FPFH features
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features = computeFPFHFeatures(cloud);
    cout << "FPFH Features computed" << endl;

    // Find the index of the most dominant feature (e.g., highest value in a specific dimension)
    int dominant_feature_index = 0;  // Adjust as needed
    float max_value = 0.0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud, *combined_cloud);


    // Iterate through FPFH features to find the most dominant feature
    for (size_t i = 0; i < fpfh_features->points.size(); ++i)
    {
        float feature_value = fpfh_features->points[i].histogram[dominant_feature_index];

        if (feature_value > max_value)
        {
            max_value = feature_value;
            dominant_feature_index = i;
        }

        pcl::PointXYZRGB p;
        p.x = cloud->points[i].x;
        p.y = cloud->points[i].y;
        p.z = cloud->points[i].z;

        if (i == dominant_feature_index)
        {
            // Assign a distinctive color to the most dominant feature point
            p.r = 255;
            p.g = 0;
            p.b = 0;
        }
        else
        {
            // Default color for other points
            p.r = 0;
            p.g = 255;
            p.b = 0;
        }

        combined_cloud->push_back(p);
    }

    // Visualize the combined point cloud with the most dominant feature
    pcl::visualization::CloudViewer viewer("Point Cloud Viewer");
    viewer.showCloud(combined_cloud);
    

    // Wait for the viewer to be closed
    while (!viewer.wasStopped())
    {
}
    return 0;
}
    