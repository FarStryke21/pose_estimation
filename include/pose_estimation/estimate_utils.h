// Create the header file for the utils.c file
#ifndef ESTIMATE_UTILS_H_
#define ESTIMATE_UTILS_H_


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

// Function to load the point cloud from the file
pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(std::string filename);

// Function to scale the point cloud
void scalePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float scale);

// Function to centre the point cloud
void centrePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

// Function to downsample the point cloud
void downsamplePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

// Function to remove the outliers from the point cloud
void removeOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

// Function to change coordinate system of the point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr change_coordinate_frame(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

// Function to compute the normals of the point cloud
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

// Function to compute the FPFH features of the point cloud where just the pointcloud is given, compute normals inside the function
pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFHFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

// Function to compute features based on curvature
pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr computePrincipalCurvatureFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

Eigen::Matrix4f computeTransformation_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2);

Eigen::Matrix4f computeTransformation_SACIA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2);

#endif /* UTILS_H_ */