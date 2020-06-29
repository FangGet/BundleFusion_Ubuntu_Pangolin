#include <iostream>
#include <string>
#include <fstream>
#include <BundleFusion.h>
#include <dirent.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <GlobalAppState.h>
#include <unistd.h>
#include <sys/time.h>

int main ( int argc, char** argv )
{
    if ( argc != 4 )
    {
        std::cout<<"usage: ./bundle_fusion_example /path/to/zParametersDefault.txt /path/to/zParametersBundlingDefault.txt /path/to/dataset"<<std::endl;
        return -1;
    }

    std::cout<<"======================BundleFusion Example=========================" << std::endl;
    std::cout<<"==  " << std::endl;
    std::cout<<"==  This is an example usage for BundleFusion SDK interface" << std::endl;
    std::cout<<"==  Author: FangGet" << std::endl;
    std::cout<<"==  " << std::endl;
    std::cout<<"===================================================================" << std::endl;

    std::string app_config_file = "";
    std::string bundle_config_file = "";
    std::string dataset_root = "";

    app_config_file = std::string ( argv[1] );
    bundle_config_file = std::string ( argv[2] );
    dataset_root = std::string ( argv[3] );

    // step 1: init all resources
    if ( !initBundleFusion ( app_config_file, bundle_config_file ) )
    {
        std::cerr<<"BundleFusion init failed, exit." << std::endl;
        return -1;
    }

    // read for bundlefusion dataset from http://graphics.stanford.edu/projects/bundlefusion/
    struct dirent *ptr;
    DIR *dir;
    dir = opendir ( dataset_root.c_str() );
    std::vector<std::string> filenames;
    while ( ( ptr=readdir ( dir ) ) !=nullptr )
    {
        if ( ptr->d_name[0]=='.' )
        {
            continue;
        }
        std::string filename_extension = std::string ( ptr->d_name );
        if ( filename_extension.size() < 10 )
            continue;
        if ( filename_extension.substr ( filename_extension.size() - 10, 10 ) == ".color.jpg" )
        {
            filenames.push_back ( filename_extension.substr ( 0, filename_extension.size() - 10 ) );
        }
    }
    std::sort ( filenames.begin(), filenames.end() );
    for ( auto& filename : filenames )
    {
        std::cout<<filename<<std::endl;
        std::string rgb_path = dataset_root + "/" + filename + ".color.jpg";
        std::string dep_path = dataset_root + "/" + filename + ".depth.png";
        //std::string pos_path = data_root + "/" + filename + ".pose.txt";
        cv::Mat rgbImage = cv::imread ( rgb_path );
        cv::Mat depthImage = cv::imread ( dep_path, cv::IMREAD_UNCHANGED );

        if ( rgbImage.empty() || depthImage.empty() )
        {
            std::cout<<"no image founded" << std::endl;
        }

         cv::imshow ( "rgb_image", rgbImage );
//         cv::imshow ( "depth_image", depthImage );
         char c = cv::waitKey ( 20 );

        if ( processInputRGBDFrame ( rgbImage,depthImage ) )
        {
            std::cout<<"\tSuccess! frame " << filename << " added into BundleFusion." << std::endl;
        }
        else
        {
            std::cout<<"\Failed! frame " << filename << " not added into BundleFusion." << std::endl;
        }
    }
    
    while(cv::waitKey (20) != 'q');


    deinitBundleFusion();

    return 0;
}
