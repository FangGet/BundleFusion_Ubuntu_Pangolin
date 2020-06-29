#ifndef __BUNDLE_FUSION_H__
#define __BUNDLE_FUSION_H__

#include <string>
#include <opencv2/opencv.hpp>

/**
 * init function for bundle fusion:
 * 1. init GPU devices
 * 2. init config file
 * 3. init rgbd sensor
 * 
 * @param app_config global app config file path
 * @param bundle_config global bundle function config file path
 * @return true if init success else false
 * */
bool initBundleFusion(std::string app_config, std::string bundle_config);

/**
 * process each input rgb and depth image
 * data may come from local dataset or different kinds of sensors(like kinect or realsense)
 * TODO we need add camera intrinsic into config file
 * @param rgb input color frame(format uint8_t * 3)
 * @param depth input depth frame(format uint16_t)
 * @return current frame process success?
 * */
bool processInputRGBDFrame(cv::Mat& rgb, cv::Mat& depth);


/**
 * release all resources include GPU, memory or other shader sources
 * */
bool deinitBundleFusion();

/**
 * 
 * set if we publish rgb color to outputwrapper
 * @param publish_flag true for publish
 * 
 * */
void setPublishRGBFlag(bool publish_flag);


/**
 * 
 * set if we publish mesh to outputwrapper
 * @param publish_flag true for publish
 * 
 * */
void setPublishMeshFlag(bool publish_flag);

/**
 * save mesh function (.ply file for meshlab)
 * @param filename save file name
 * @param overwriteExistingFile existing file will be replaced if true
 * @return save success
 * */
bool saveMeshIntoFile ( const std::string& filename, bool overwriteExistingFile /*= false*/ );

#endif
