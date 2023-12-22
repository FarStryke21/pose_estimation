// Utility functions for the pose estimation

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
#include <pcl/features/principal_curvatures.h>
// #include "estimate_utils.h"

// Function to load the point cloud from the file
pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(std::string filename)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, *cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
    }
    return cloud;
}

// Function to change the origin of the point cloud to the centre of mass
void centrePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Calculate the centroid of the pointcloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Create a transformation matrix to move the pointcloud to the origin
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 3) = -centroid[0];
    transform(1, 3) = -centroid[1];
    transform(2, 3) = -centroid[2];

    // Apply this trabsformation to the target cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

    *cloud = *transformed_cloud;
    cout << "Point cloud centred" << endl;
}

// Function to downsample the point cloud
void downsamplePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(1, 1, 1);
    sor.filter(*cloud_filtered);

    // put the filtered cloud in the original cloud
    *cloud = *cloud_filtered;
    cout << "Point cloud downsampled" << endl;
}

// Fucntion to scale the point cloud
void scalePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float scale)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scaled(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < cloud->points.size(); i++)
    {
        cloud->points[i].x *= scale;
        cloud->points[i].y *= scale;
        cloud->points[i].z *= scale;
    }
    cout << "Point cloud scaled by " << scale << endl;
}

// Function to remove the outliers from the point cloud
void removeOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);
    // put the filtered cloud in the original cloud
    *cloud = *cloud_filtered;
    cout << "Outliers removed from the point cloud" << endl;
}

// Function to compute the normals of the point cloud
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(5);
    ne.compute(*normals);
    return normals;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr change_coordinate_frame(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // Calculate the centroid of the pointcloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Create a transformation matrix to move the pointcloud to the origin
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 3) = -centroid[0];
    transform(1, 3) = -centroid[1];
    transform(2, 3) = -centroid[2];

    // Apply this trabsformation to the target cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

    std::cout << "Aligned Coordinate Frames ..." << std::endl;
    return transformed_cloud;
}


// Function to compute the FPFH features of the point cloud where just the pointcloud is given, compute normals inside the function
pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFHFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(computeNormals(cloud));
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(5);
    fpfh.compute(*fpfhs);
    return fpfhs;
}

// Function to compute the SHOT features of the point cloud where just the pointcloud is given, compute normals inside the function
pcl::PointCloud<pcl::SHOT352>::Ptr computeSHOTFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::SHOT352>::Ptr shots(new pcl::PointCloud<pcl::SHOT352>());
    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    shot.setInputCloud(cloud);
    shot.setInputNormals(computeNormals(cloud));
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    shot.setSearchMethod(tree);
    shot.setRadiusSearch(0.05);
    shot.compute(*shots);
    return shots;
}

// Function to compute the Features based on Principal Curvature of the point cloud where just the pointcloud is given
pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr computePrincipalCurvatureFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvature;
    principal_curvature.setInputCloud(cloud);
    principal_curvature.setInputNormals(computeNormals(cloud));
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    principal_curvature.setSearchMethod(tree);
    principal_curvature.setRadiusSearch(5);
    principal_curvature.compute(*principal_curvatures);
    return principal_curvatures;
}
//Function to compute USC features of the point cloud where just the pointcloud is given, compute normals inside the function
// pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr computeUSCFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
// {
//     pcl::PointCloud<pcl::UniqueShapeContext1960>::Ptr uscs(new pcl::PointCloud<pcl::UniqueShapeContext1960>());
//     pcl::UniqueShapeContext<pcl::PointXYZ, pcl::UniqueShapeContext1960, pcl::ReferenceFrame> usc;
//     usc.setInputCloud(cloud);
//     usc.setInputNormals(computeNormals(cloud));
//     pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//     usc.setSearchMethod(tree);
//     usc.setRadiusSearch(0.05);
//     usc.compute(*uscs);
//     return uscs;
// }

// Function to compute the transformation between the two point clouds
Eigen::Matrix4f computeTransformation_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2)
{
    std::cout << "\nICP Starting ... " << std::endl;
    clock_t start, end;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud1);
    icp.setInputTarget(cloud2);

    icp.setTransformationEpsilon(1e-8);

    pcl::PointCloud<pcl::PointXYZ> Final;
    start = clock();
    icp.align(Final);
    end = clock();
    std::cout << "Time required : " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

    if (icp.hasConverged())
    {
        std::cout << "ICP converged with a score of " << icp.getFitnessScore() << std::endl;
        // std::cout << "Transformation matrix:\n" << icp.getFinalTransformation() << std::endl;
        return icp.getFinalTransformation();
    }
    else
    {
        std::cerr << "ICP did not converge" << std::endl;
        return Eigen::Matrix4f::Identity();
    }
}

// Function to compute the transformation between the two point clouds using RANSAC
Eigen::Matrix4f computeTransformation_SACIA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2)
{
    std::cout << "\nSAC-IA Starting ... " << std::endl;
    clock_t start, end;

    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::PrincipalCurvatures> sac_ia;
    sac_ia.setInputSource(cloud1);
    sac_ia.setSourceFeatures(computePrincipalCurvatureFeatures(cloud1));
    sac_ia.setInputTarget(cloud2);
    sac_ia.setTargetFeatures(computePrincipalCurvatureFeatures(cloud2));

    // sac_ia.setMaximumIterations(500); // Adjust as needed
    // sac_ia.setDistanceThreshold(0.02); // Adjust as needed
    // sac_ia.setMaxCorrespondenceDistance(1000.0);
    // sac_ia.setMinSampleDistance (0.5);
    

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    start = clock();
    sac_ia.align(*aligned_cloud);
    end = clock();
    std::cout << "Time required : " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

    if (sac_ia.hasConverged())
    {
        std::cout << "SAC-IA converged with a score of " << sac_ia.getFitnessScore() << std::endl;
        // std::cout << "Transformation matrix:\n" << sac_ia.getFinalTransformation() << std::endl;
        cloud1->clear();
        *cloud1 = *aligned_cloud;

        return sac_ia.getFinalTransformation();
    }
    else
    {
        std::cerr << "SAC-IA did not converge" << std::endl;
        return Eigen::Matrix4f::Identity();
    }
}

// Function to visualize the point clouds received in a list
void visualizePointClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds)
{
    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addCoordinateSystem(0.1);
    viewer.initCameraParameters();
    for (int i = 0; i < clouds.size(); i++)
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(clouds[i], 0, 255, 0);
        viewer.addPointCloud<pcl::PointXYZ>(clouds[i], single_color, "cloud" + std::to_string(i));
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud" + std::to_string(i));
    }
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}

