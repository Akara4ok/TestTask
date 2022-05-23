#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


void readFromBinary(std::string path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
pcl::PointXYZ findCenter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void normalizedCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
void projectToPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr projectedCloud, Eigen::Vector3d axis);
Eigen::MatrixXd getRotationMatrix(Eigen::Vector3d dir);
void rotatePlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::Vector3d dir);
void Draw(int width, int height, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string path, float alpha = 50);


int main (int argc, char**argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if(argc != 2)
    {
        std::cout << "Wrong numbers of arguments" << "\n";
        return -1;
    }

    try
    {
        readFromBinary(argv[1], cloud);
    }
    catch (std::runtime_error e)
    {
        std::cout << e.what();
        return -1;
    }

    pcl::PointXYZ center = findCenter(cloud);
    std::cout << center << "\n";

    normalizedCloud(cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr projectedCloud(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Vector3d dir1(1, -1, 1);
    projectToPlane(cloud, projectedCloud, dir1); // projecting cloud to plane, where dir is normal vector to plane
    rotatePlane(projectedCloud, dir1); // rotate projected cloud to plane which is parallel to XY
    Draw(1920, 1080, projectedCloud, "img1.jpg"); // Draw the result

    Eigen::Vector3d dir2(0, 0, 1);
    projectToPlane(cloud, projectedCloud, dir2);
    rotatePlane(projectedCloud, dir2);

    Draw(1920, 1080, projectedCloud, "img2.jpg");

    return 0;
}



void readFromBinary(std::string path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::ifstream fin;
    fin.open(path, std::fstream::in | std::fstream::binary);
    if(!fin.is_open())
        throw std::runtime_error("Error! Unable to open the file.\n");

    pcl::PointXYZ point;
    while (!fin.eof())
    {
        fin.read(reinterpret_cast<char*>(&point), 3 * sizeof(float));
        cloud->points.push_back(point);
    }
    fin.close();
}

pcl::PointXYZ findCenter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointXYZ center(0, 0, 0);
    for (const auto& point : cloud->points) //to find center we need to sum all points and find mean
    {
        center.x = center.x + point.x;
        center.y = center.y + point.y;
        center.z = center.z + point.z;
    }
    center.x = center.x / cloud->points.size();
    center.y = center.y / cloud->points.size();
    center.z = center.z / cloud->points.size();
    return center;
}

void normalizedCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointXYZ center = findCenter(cloud); 
    // to match center of points and center of coordinates, 
    // we need to find vector between this centers, and translate all points due this vector
    for (auto& point : cloud->points)
    {
        point.x = point.x - center.x;
        point.y = point.y - center.y;
        point.z = point.z - center.z;
    }
}

void projectToPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr projectedCloud, Eigen::Vector3d axis)
{
    // setting coeffs of the plane to project points
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    coefficients->values.resize (4);
    coefficients->values[0] = axis(0);
    coefficients->values[1] = axis(1);
    coefficients->values[2] = axis(2);
    coefficients->values[3] = 0;

    // Create the filtering object
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (cloud);
    proj.setModelCoefficients (coefficients);
    proj.filter (*projectedCloud);
}

Eigen::MatrixXd getRotationMatrix(Eigen::Vector3d dir)
{
    Eigen::MatrixXd result(3, 3);

    dir.normalize();
    // we need to align direction vector with z-axis, so we need to normalise this vector, 
    // find normal vector to plane created by dir vector and z-axis
    // and rotate by angle between dir and z=axis
    float cosA = dir(2);
    Eigen::Vector3d axis = dir.cross(Eigen::Vector3d(0, 0, 1));
    float sinA = sqrt(1 - cosA * cosA);

    // the rotation matrix custom axis and angle
    result << cosA + (1 - cosA) * axis(0) * axis(0),
        (1 - cosA) * axis(0) * axis(1) - sinA * axis(2),
        (1 - cosA) * axis(0) * axis(2) + sinA * axis(1),

        (1 - cosA) * axis(1) * axis(0) + sinA * axis(2),
        cosA + (1 - cosA) * axis(1) * axis(1),
        (1 - cosA) * axis(1) * axis(2) - sinA * axis(0),

        (1 - cosA) * axis(2) * axis(0) - sinA * axis(1),
        (1 - cosA) * axis(2) * axis(1) + sinA * axis(0),
        cosA + (1 - cosA) * axis(2) * axis(2);

    return result;
}

void rotatePlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Eigen::Vector3d dir)
{
    Eigen::MatrixXd result = getRotationMatrix(dir);
    // rotate all points by multiplication of rotate matrix and point
    for (auto& point : cloud->points)
    {
        Eigen::MatrixXd vector(3, 1);
        vector << point.x, point.y,  point.z;
        Eigen::MatrixXd rotatedVector(3, 1);
        rotatedVector = result * vector;
        point.x = rotatedVector(0);
        point.y = rotatedVector(1);
        point.z = rotatedVector(2);
    }
}

void Draw(int width, int height, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string path, float alpha)
{
    // because of in our coordinate system (0, 0) is center, but (0, 0) is up left corner in opencv, we need to move all points
    cv::Point2f toCorner(width / 2, height / 2);
    cv::Mat Image(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));
    
    int count = 0;
    
    for (const auto& point : cloud->points)
    {
        cv::Point2f p;
        p.x = point.x;
        p.y = - point.y; // in opencv increasing row number we move down, so wee need put minus 
        cv::circle(Image,
            p * alpha + toCorner, //set parametr alpha to increase distance between points
            1,
            cv::Scalar(255, 255, 255),
            cv::FILLED);
    }
    cv::imwrite(path, Image);
}