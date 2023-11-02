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
    // icp.setMaximumIterations(10);
    icp.setTransformationEpsilon(1e-8);

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
        // pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::transformPointCloud(*target_cloud, *transformed_cloud, icp.getFinalTransformation());

        // viewer.addPointCloud(transformed_cloud, "Transformed");
        // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Transformed");
        viewer.addPointCloud(target_cloud, "Target");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Target");
        // viewer.addPointCloud(source_cloud, "Source");
        // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Source");
        viewer.setBackgroundColor(0.0, 0.0, 0.0);
        viewer.addCoordinateSystem(50.0);

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
    ne.setRadiusSearch(5);

    // Compute the features
    ne.compute(*normals);

    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);

    // Use all neighbors in a sphere of radius
    pcl::search::KdTree<pcl::PointXYZ>::Ptr fpfh_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod(fpfh_tree);
    fpfh.setRadiusSearch(5);

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
    std::cout << "Source Features : " << features_source->size() << std::endl;
    sac_ia.setInputTarget(target_cloud);
    sac_ia.setTargetFeatures(features_target);
    std::cout << "Target Features : " << features_target->size() << std::endl;

    // Set the number of iterations and distance threshold
    // sac_ia.setMaximumIterations(500); // Adjust as needed
    // sac_ia.setDistanceThreshold(0.02); // Adjust as needed
    // sac_ia.setMaxCorrespondenceDistance(1000.0);
    // sac_ia.setMinSampleDistance (0.5);
    std::cout << "Max Correspondence Distance : " << sac_ia.getMaxCorrespondenceDistance() << std::endl;
    std::cout << "Min Sample Distance : " << sac_ia.getMinSampleDistance() << std::endl;
    std::cout << "Max Iterations : " << sac_ia.getMaximumIterations() << std::endl;

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
        //print how many iterations it took to converge

        // // Visualize the aligned cloud*
        // pcl::visualization::PCLVisualizer viewer("SAC-IA Result");
        // viewer.addPointCloud(aligned_cloud, "Aligned");

        // // Use the transformation matrix to transform the target cloud
        // // pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // // pcl::transformPointCloud(*target_cloud, *transformed_cloud, sac_ia.getFinalTransformation());

        // // viewer.addPointCloud(transformed_cloud, "Transformed");
        // // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Transformed");
        // // viewer.addPointCloud(target_cloud, "Target");
        // // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Target");

        // viewer.addPointCloud(source_cloud, "Source");
        // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "Source");
        // viewer.setBackgroundColor(0.0, 0.0, 0.0);
        // // Display global reference frame
        // viewer.addCoordinateSystem(50.0);

        // while (!viewer.wasStopped())
        // {
        //     viewer.spinOnce();
        // }
        // // Save the target cloud as a PLY file
        // pcl::io::savePLYFile("owl_aligned.ply", *aligned_cloud);
        // pcl::io::savePLYFile("owl_target.ply", *target_cloud);
    }
    else
    {
        std::cerr << "SAC-IA did not converge" << std::endl;
    }

    return aligned_cloud;
}  

// write a function that takes in a pointcloud and changes its coordinate frame to be at the centroid pf the pointcloud
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

    std::cout << "Brought Pointcloud to centroid ..." << std::endl;
    return transformed_cloud;
}


int main(int argc, char** argv)
{
    // Load the source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPLYFile("owl_sampled.ply", *target_cloud);        // Base File
    pcl::io::loadPLYFile("owl_scan_1.ply", *source_cloud);   // Scan File
    std::cout << "Loaded " << std::endl;

    // // Create a random transformation matrix and apply it to the target cloud
    // Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    // float theta = M_PI / 4; // The angle of rotation in radians
    // transform(0, 0) = cos(theta);
    // transform(0, 1) = -sin(theta);
    // transform(1, 0) = sin(theta);
    // transform(1, 1) = cos(theta);
    // transform(0, 3) = 25; // The translation on x axis
    // transform(1, 3) = 10; // The translation on y axis
    // transform(2, 3) = 15; // The translation on z axis
    // // Apply this trabsformation to the target cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::transformPointCloud(*target_cloud, *transformed_cloud, transform);
    // *target_cloud = *transformed_cloud;

    // Magnify pointcloud by 1000 times (Use this when target cloud is from thge scannner)
    float magnification_factor = 1000.0;
    for (size_t i = 0; i < source_cloud->points.size(); ++i) {
        source_cloud->points[i].x *= magnification_factor;
        source_cloud->points[i].y *= magnification_factor;
        source_cloud->points[i].z *= magnification_factor;
    }
    std::cout << "Magnified " << std::endl;

    // ----------------------Downsample the target cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(source_cloud); 
    voxel_grid.setLeafSize(1, 1, 1);
    voxel_grid.filter(*downsampled_target_cloud);
    *source_cloud = *downsampled_target_cloud;
    std::cout << "Downsampled " << std::endl;

    // Remove outliers from the target cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(source_cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*filtered_target_cloud);
    *source_cloud = *filtered_target_cloud;
    std::cout << "Filtered " << std::endl;

    // Change the coordinate frame of the target cloud to be at the origin
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    transformed_target_cloud = change_coordinate_frame(source_cloud);
    *source_cloud = *transformed_target_cloud;

    // Call the Pose Estimate function
    // pose_estimate_ICP(source_cloud, target_cloud);
    source_cloud = pose_estimate_SACIA(source_cloud, target_cloud); 
    pose_estimate_ICP(source_cloud, target_cloud);

    return 0;
}
