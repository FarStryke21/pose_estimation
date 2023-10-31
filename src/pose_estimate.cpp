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


pcl::PointCloud<pcl::PointXYZ>::Ptr pose_estimate_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud)
{
    // ---------------------Initialize the ICP object
    // Calculate the time required to estimate the pose
    clock_t start, end;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);

    // Set termination criteria
    icp.setMaximumIterations(10);
    // icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1);

    // Align the clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    start = clock();
    std::cout << "ICP Starting ... " << std::endl;
    icp.align(*aligned_cloud);

    end = clock();
    std::cout << "Time required : " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

    if (icp.hasConverged())
    {
        std::cout << "ICP converged with a score of " << icp.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:\n" << icp.getFinalTransformation() << std::endl;

        // Visualize the aligned cloud
        pcl::visualization::PCLVisualizer viewer("ICP Result");
        viewer.addPointCloud(aligned_cloud, "Aligned");

        // Use the transformation matrix to transform the target cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*target_cloud, *transformed_cloud, icp.getFinalTransformation());

        viewer.addPointCloud(transformed_cloud, "Transformed");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Transformed");
        // viewer.addPointCloud(target_cloud, "Target");
        // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Target");
        // viewer.addPointCloud(source_cloud, "Source");
        // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "Source");
        viewer.setBackgroundColor(0.0, 0.0, 0.0);

        while (!viewer.wasStopped())
        {
            viewer.spinOnce();
        }
    }
    else
    {
        std::cerr << "ICP did not converge" << std::endl;
    }

}

void computeFeatures(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                     pcl::PointCloud<pcl::FPFHSignature33>::Ptr& features)
{
    // Create a Normal Estimation object
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // Create an empty KdTree for neighbor search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree);

    // Output dataset (normals)
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 
    ne.setRadiusSearch(0.05);

    // Compute the features
    ne.compute(*normals);

    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);

    // Use all neighbors in a sphere of radius 0.2m
    pcl::search::KdTree<pcl::PointXYZ>::Ptr fpfh_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod(fpfh_tree);
    fpfh.setRadiusSearch(0.05);

    // Compute the features
    features.reset(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh.compute(*features);
}

// Write a function that uses SAC-IA to estimate the pose
pcl::PointCloud<pcl::PointXYZ>::Ptr pose_estimate_SACIA(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud)
{   
    clock_t start, end;
    // Feature estimation
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_source(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_target(new pcl::PointCloud<pcl::FPFHSignature33>);

    computeFeatures(source_cloud, features_source);
    computeFeatures(target_cloud, features_target);

    // ---------------------Initialize the SAC-IA object
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(source_cloud);
    sac_ia.setSourceFeatures(features_source);
    sac_ia.setInputTarget(target_cloud);
    sac_ia.setTargetFeatures(features_target);

    // Set the number of iterations and distance threshold
    // sac_ia.setMaximumIterations(100); // Adjust as needed
    // sac_ia.setDistanceThreshold(0.02); // Adjust as needed

    // Align the clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::cout << "SAC-IA Starting ... " << std::endl;
    start = clock();
    sac_ia.align(*aligned_cloud);
    end = clock();
    std::cout << "Time required : " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

    
    if (sac_ia.hasConverged())
    {
        std::cout << "SAC-IA converged with a score of " << sac_ia.getFitnessScore() << std::endl;
        std::cout << "Transformation matrix:\n" << sac_ia.getFinalTransformation() << std::endl;

        // Visualize the aligned cloud
        pcl::visualization::PCLVisualizer viewer("SAC-IA Result");
        viewer.addPointCloud(aligned_cloud, "Aligned");

        // Use the transformation matrix to transform the target cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*target_cloud, *transformed_cloud, sac_ia.getFinalTransformation());

        viewer.addPointCloud(transformed_cloud, "Transformed");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Transformed");
        viewer.addPointCloud(target_cloud, "Target");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Target");
        // viewer.addPointCloud(source_cloud, "Source");
        // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "Source");
        viewer.setBackgroundColor(0.0, 0.0, 0.0);

        while (!viewer.wasStopped())
        {
            viewer.spinOnce();
        }
    }
    else
    {
        std::cerr << "SAC-IA did not converge" << std::endl;
    }
}    


int main(int argc, char** argv)
{
    // Load the source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPLYFile("owl_sampled.ply", *source_cloud);        // Base File
    pcl::io::loadPLYFile("owl_sampled_modified.ply", *target_cloud);   // Scan File
    std::cout << "Loaded " << std::endl;

    // Create a random transformation matrix and apply it to the target cloud
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    float theta = M_PI / 4; // The angle of rotation in radians
    transform(0, 3) = 25; // The translation on x axis
    transform(1, 3) = 10.0; // The translation on y axis
    transform(2, 3) = 15; // The translation on z axis
    // Apply this trabsformation to the target cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*target_cloud, *transformed_cloud, transform);
    *target_cloud = *transformed_cloud;


    // Magnify pointcloud by 1000 times
    // float magnification_factor = 1000.0;
    // for (size_t i = 0; i < target_cloud->points.size(); ++i) {
    //     target_cloud->points[i].x *= magnification_factor;
    //     target_cloud->points[i].y *= magnification_factor;
    //     target_cloud->points[i].z *= magnification_factor;
    // }

    // ----------------------Downsample the target cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    // voxel_grid.setInputCloud(target_cloud); 
    // voxel_grid.setLeafSize(2, 2, 2);
    // voxel_grid.filter(*downsampled_target_cloud);
    // *target_cloud = *downsampled_target_cloud;

    // // Remove outliers from the target cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    // sor.setInputCloud(target_cloud);
    // sor.setMeanK(50);
    // sor.setStddevMulThresh(1.0);
    // sor.filter(*filtered_target_cloud);
    // *target_cloud = *filtered_target_cloud;

    // Call the Pose Estimate function
    // pose_estimate_ICP(source_cloud, target_cloud);
    pose_estimate_SACIA(source_cloud, target_cloud); 

    return 0;
}
