#include <iostream>
#include <cmath>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

#include "pose_estimation/estimate_utils.h"

// Define a point cloud structure
struct Point 
{
    double x, y, z;
    Point(double x, double y, double z) : x(x), y(y), z(z) {}
};

// Function to convert pcl::PointCloud to vector<Point>
std::vector<Point> pclToVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pclCloud) 
{
    std::vector<Point> cloud;
    for (const auto& point : pclCloud->points) 
    {
        cloud.emplace_back(point.x, point.y, point.z);
    }
    return cloud;
}

// Function to convert vector<Point> to pcl::PointCloud
pcl::PointCloud<pcl::PointXYZ>::Ptr vectorToPCL(const std::vector<Point>& cloud) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pclCloud->width = cloud.size();
    pclCloud->height = 1;
    pclCloud->points.resize(pclCloud->width * pclCloud->height);
    for (size_t i = 0; i < cloud.size(); ++i) 
    {
        pclCloud->points[i].x = cloud[i].x;
        pclCloud->points[i].y = cloud[i].y;
        pclCloud->points[i].z = cloud[i].z;
    }
    return pclCloud;
}

// Function to find correspondences using nearest neighbor search
std::vector<int> findCorrespondences(const pcl::PointCloud<pcl::PointXYZ>::Ptr& sourceCloud,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr& targetCloud) 
{
    std::vector<int> correspondences;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(targetCloud);

    for (size_t i = 0; i < sourceCloud->points.size(); ++i) 
    {
        pcl::PointXYZ searchPoint = sourceCloud->points[i];
        std::vector<int> pointIndices(1);
        std::vector<float> pointDistances(1);

        if (kdtree.nearestKSearch(searchPoint, 1, pointIndices, pointDistances) > 0) 
        {
            // Correspondence found, add index to the list
            correspondences.push_back(pointIndices[0]);
        }
    }

    return correspondences;
}

// Function to calculate the transformation matrix
Eigen::Matrix4f calculateTransformationMatrix(const pcl::PointCloud<pcl::PointXYZ>::Ptr& sourceCloud,
                                              const pcl::PointCloud<pcl::PointXYZ>::Ptr& targetCloud,
                                              const std::vector<int>& correspondences) 
{
    // Calculate centroids
    pcl::PointXYZ centroidSource, centroidTarget;
    for (int index : correspondences) 
    {
        centroidSource.x += sourceCloud->points[index].x;
        centroidSource.y += sourceCloud->points[index].y;
        centroidSource.z += sourceCloud->points[index].z;

        centroidTarget.x += targetCloud->points[index].x;
        centroidTarget.y += targetCloud->points[index].y;
        centroidTarget.z += targetCloud->points[index].z;
    }

    centroidSource.x /= correspondences.size();
    centroidSource.y /= correspondences.size();
    centroidSource.z /= correspondences.size();

    centroidTarget.x /= correspondences.size();
    centroidTarget.y /= correspondences.size();
    centroidTarget.z /= correspondences.size();

    // Center the point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr centeredSource(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr centeredTarget(new pcl::PointCloud<pcl::PointXYZ>);

    for (int index : correspondences) 
    {
        pcl::PointXYZ centeredPointSource = sourceCloud->points[index];
        centeredPointSource.x -= centroidSource.x;
        centeredPointSource.y -= centroidSource.y;
        centeredPointSource.z -= centroidSource.z;
        centeredSource->points.push_back(centeredPointSource);

        pcl::PointXYZ centeredPointTarget = targetCloud->points[index];
        centeredPointTarget.x -= centroidTarget.x;
        centeredPointTarget.y -= centroidTarget.y;
        centeredPointTarget.z -= centroidTarget.z;
        centeredTarget->points.push_back(centeredPointTarget);
    }

    // Compute the covariance matrix
    Eigen::Matrix3f covarianceMatrix = Eigen::Matrix3f::Zero();

    for (size_t i = 0; i < centeredSource->points.size(); ++i) 
    {
        const pcl::PointXYZ& centeredPointSource = centeredSource->points[i];
        const pcl::PointXYZ& centeredPointTarget = centeredTarget->points[i];

        covarianceMatrix(0, 0) += centeredPointSource.x * centeredPointTarget.x;
        covarianceMatrix(0, 1) += centeredPointSource.x * centeredPointTarget.y;
        covarianceMatrix(0, 2) += centeredPointSource.x * centeredPointTarget.z;
        covarianceMatrix(1, 0) += centeredPointSource.y * centeredPointTarget.x;
        covarianceMatrix(1, 1) += centeredPointSource.y * centeredPointTarget.y;
        covarianceMatrix(1, 2) += centeredPointSource.y * centeredPointTarget.z;
        covarianceMatrix(2, 0) += centeredPointSource.z * centeredPointTarget.x;
        covarianceMatrix(2, 1) += centeredPointSource.z * centeredPointTarget.y;
        covarianceMatrix(2, 2) += centeredPointSource.z * centeredPointTarget.z;
    
    }

    covarianceMatrix /= centeredSource->points.size();

    // Perform Singular Value Decomposition
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f rotationMatrix = svd.matrixU() * svd.matrixV().transpose();

    // Calculate translation vector
    Eigen::Vector3f translationVector = Eigen::Vector3f(centroidTarget.x, centroidTarget.y, centroidTarget.z) -
                                        rotationMatrix * Eigen::Vector3f(centroidSource.x, centroidSource.y, centroidSource.z);

    // Compose the transformation matrix
    Eigen::Matrix4f transformationMatrix = Eigen::Matrix4f::Identity();
    transformationMatrix.block<3, 3>(0, 0) = rotationMatrix;
    transformationMatrix.block<3, 1>(0, 3) = translationVector;

    return transformationMatrix;
}

// Function to calculate the error between two point clouds
double calculateError(const pcl::PointCloud<pcl::PointXYZ>::Ptr& alignedSource,
                       const pcl::PointCloud<pcl::PointXYZ>::Ptr& targetCloud) 
{
    double error = 0.0;

    for (size_t i = 0; i < alignedSource->points.size(); ++i) 
    {
        const pcl::PointXYZ& sourcePoint = alignedSource->points[i];
        const pcl::PointXYZ& targetPoint = targetCloud->points[i];

        double distance = std::sqrt(
            std::pow(sourcePoint.x - targetPoint.x, 2) +
            std::pow(sourcePoint.y - targetPoint.y, 2) +
            std::pow(sourcePoint.z - targetPoint.z, 2)
        );

        error += distance;
    }

    return error / alignedSource->points.size();
}

// Custom ICP implementation for PCL point clouds
void customICP(const pcl::PointCloud<pcl::PointXYZ>::Ptr& sourceCloud,
               const pcl::PointCloud<pcl::PointXYZ>::Ptr& targetCloud,
               int maxIterations, double tolerance) 
{
    Eigen::Matrix4f transform_1 = computeTransformation_SACIA(sourceCloud, targetCloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr alignedSource(new pcl::PointCloud<pcl::PointXYZ>());
    *alignedSource = *sourceCloud;

    // Print the length of the point clouds
    std::cout << "Source Cloud Size: " << alignedSource->points.size() << std::endl;

    for (int iteration = 0; iteration < maxIterations; ++iteration) 
    {
        std::cout << "Iteration " << iteration + 1 << std::endl;
        // Step 1: Find correspondences (matching points) between source and target
        std::vector<int> correspondences = findCorrespondences(alignedSource, targetCloud);
        std::cout << "Found " << correspondences.size() << " correspondences.\n";

        // Step 2: Calculate the transformation
        Eigen::Matrix4f transformationMatrix = calculateTransformationMatrix(alignedSource, targetCloud, correspondences);
        std::cout << "Transformation Matrix:\n" << transformationMatrix << std::endl;

        // Apply the transformation to the source cloud
        pcl::transformPointCloud(*alignedSource, *alignedSource, transformationMatrix);

        // Step 3: Check for convergence
        double error = calculateError(alignedSource, targetCloud);
        std::cout << "Error: " << error << std::endl;

        if (error < tolerance) 
        {
            std::cout << "Converged after " << iteration + 1 << " iterations.\n";
            break;
        }
    }

    // Output or use the alignedSource point cloud
    pcl::visualization::PCLVisualizer viewer("NICP Result");
    viewer.addPointCloud(targetCloud, "Target Cloud");
    viewer.addPointCloud(alignedSource, "Aligned Source Cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Target Cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Aligned Source Cloud");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addCoordinateSystem(25.0);
    viewer.setCameraPosition(0, 0, -1, 0, -1, 0, 0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
    viewer.close();
}

int main() {
    // Load sample point clouds using PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Load or generate your point clouds here using PCL functions
    targetCloud = loadPointCloud("data/owl_sampled.ply");
    sourceCloud = loadPointCloud("data/owl_target.ply");

    // Set ICP parameters
    int maxIterations = 50;
    double tolerance = 1e-6;

    // Perform custom ICP
    customICP(sourceCloud, targetCloud, maxIterations, tolerance);

    return 0;
}
