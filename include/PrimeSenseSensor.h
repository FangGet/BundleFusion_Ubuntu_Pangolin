#pragma once

/************************************************************************/
/* Prime sense depth camera: Warning this is highly untested atm        */
/************************************************************************/

#include "GlobalAppState.h"

//Only working with OpenNI 2 SDK
#ifdef OPEN_NI

#include "RGBDSensor.h"
//#include <OpenNI.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <list>

class PrimeSenseSensor : public RGBDSensor
{
public:

	//! Constructor; allocates CPU memory and creates handles
	PrimeSenseSensor();

	//! Destructor; releases allocated ressources
	~PrimeSenseSensor();

	//! Initializes the sensor
	void createFirstConnected();

	//! Processes the depth data (and color)
	bool processDepth();
	
	bool receiveDepthAndColor(cv::Mat& rgb, cv::Mat& depth);
	
	std::string getSensorName() const {
		return "orbbec";
	}
	//! Processes the Kinect color data
	bool processColor()
	{
		return true;
	}

protected:
	//! reads depth and color from the sensor
	bool readDepthAndColor(const cv::Mat& rgbTest, const cv::Mat& depthTest,float* depthFloat, vec4uc* colorRGBX);

	int				m_picNum = 30;//  30;//4000;//6000;//11000;//6830;
	// to prevent drawing until we have data for both streams
	bool			m_bDepthReceived;
	bool			m_bColorReceived;

	bool			m_bDepthImageIsUpdated;
	bool			m_bDepthImageCameraIsUpdated;
	bool			m_bNormalImageCameraIsUpdated;

	bool			m_kinect4Windows;
// #ifndef MY_TEST_IMAGE
// 	openni::VideoMode			m_depthVideoMode;
// 	openni::VideoMode			m_colorVideoMode;
// 
// 
// 	openni::VideoFrameRef		m_depthFrame;
// 	openni::VideoFrameRef		m_colorFrame;
// 
// 	openni::Device				m_device;
// 	openni::VideoStream			m_depthStream;
// 	openni::VideoStream			m_colorStream;
// 	openni::VideoStream**		m_streams;
// #endif
	//cv::Mat						rgbTest;
	//cv::Mat						depthTest;
	
};

#endif
