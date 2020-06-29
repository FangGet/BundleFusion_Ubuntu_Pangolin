
#include "stdafx.h"

#include "PrimeSenseSensor.h"

PrimeSenseSensor::PrimeSenseSensor()
{
    m_bDepthReceived = false;
    m_bColorReceived = false;

    m_bDepthImageIsUpdated = false;
    m_bDepthImageCameraIsUpdated = false;
    m_bNormalImageCameraIsUpdated = false;
}

PrimeSenseSensor::~PrimeSenseSensor()
{
}

void PrimeSenseSensor::createFirstConnected()
{

    RGBDSensor::init ( 640, 480, 640, 480, 1 );
    initializeDepthIntrinsics ( GlobalAppState::get().s_cameraIntrinsicFx,
                                GlobalAppState::get().s_cameraIntrinsicFy,
                                GlobalAppState::get().s_cameraIntrinsicCx,
                                GlobalAppState::get().s_cameraIntrinsicCy );

    initializeColorIntrinsics ( GlobalAppState::get().s_cameraIntrinsicFx,
                                GlobalAppState::get().s_cameraIntrinsicFy,
                                GlobalAppState::get().s_cameraIntrinsicCx,
                                GlobalAppState::get().s_cameraIntrinsicCy );

    initializeColorExtrinsics ( mat4f::identity() );
    initializeDepthExtrinsics ( mat4f::identity() );


}

bool PrimeSenseSensor::receiveDepthAndColor ( cv::Mat& rgb, cv::Mat& depth )
{
    bool hr = true;

    m_bDepthImageIsUpdated = false;
    m_bDepthImageCameraIsUpdated = false;
    m_bNormalImageCameraIsUpdated = false;

    hr = readDepthAndColor ( rgb,depth, getDepthFloat(), m_colorRGBX );

    m_bDepthImageIsUpdated = true;
    m_bDepthImageCameraIsUpdated = true;
    m_bNormalImageCameraIsUpdated = true;

    m_bDepthReceived = true;
    m_bColorReceived = true;

    return hr;
}


bool PrimeSenseSensor::processDepth()
{

    bool hr = true;
    return hr;
}



bool PrimeSenseSensor::readDepthAndColor ( const cv::Mat& rgbTest, const cv::Mat& depthTest,float* depthFloat, vec4uc* colorRGBX )
{
    bool hr = true;

    if ( rgbTest.empty() )
    {
        std::cout << "no rgb!" << std::endl;
        hr = false;
    }
    if ( depthTest.empty() )
    {
        std::cout << "no depth!" << std::endl;
        hr = false;
    }

    if ( rgbTest.empty() || depthTest.empty() )
    {
        return false;
    }

    const uint16_t* pDepth = ( const uint16_t* ) depthTest.data;
    const uint8_t* pImage = ( const uint8_t* ) rgbTest.data;

    if ( !depthTest.empty() && !rgbTest.empty() )
    {
        unsigned int width = depthTest.cols;
        unsigned int nPixels = depthTest.cols * depthTest.rows;

        for ( unsigned int i = 0; i < nPixels; i++ )
        {
            const int x = i%width;
            const int y = i / width;
            const int src = y*width + ( width - 1 - x );
            const uint16_t& p = pDepth[src];

            float dF = ( float ) p*0.001f;
            if ( dF >= GlobalAppState::get().s_sensorDepthMin && dF <= GlobalAppState::get().s_sensorDepthMax )
            {
                depthFloat[i] = dF;
            }
            else
            {
                depthFloat[i] = -std::numeric_limits<float>::infinity();
            }
        }
        incrementRingbufIdx();
    }

    // check if we need to draw depth frame to texture
    //if (m_depthFrame.isValid() && m_colorFrame.isValid())
    if ( !depthTest.empty() && !rgbTest.empty() )
    {
        //unsigned int width = m_colorFrame.getWidth();
        //unsigned int height = m_colorFrame.getHeight();
        unsigned int width = rgbTest.cols;
        unsigned int height = rgbTest.rows;
        unsigned int nPixels = width*height;

        for ( unsigned int i = 0; i < nPixels; i++ )
        {
            const int x = i%width;
            const int y = i / width;

            int y2 = 0;
            if ( m_colorWidth == 1280 )
            {
                y2 = y + 64 / 2 - 10 - ( unsigned int ) ( ( ( float ) y / ( ( float ) ( height - 1 ) ) ) * 64 + 0.5f );
            }
            else
            {
                y2 = y;
            }

            if ( y2 >= 0 && y2 < ( int ) height )
            {
                unsigned int Index1D = y2*width + ( width - 1 - x );	//x-flip here

                //const openni::RGB888Pixel& pixel = pImage[Index1D];

                unsigned int c = 0;
                c |= pImage[3*Index1D + 0];
                c <<= 8;
                c |= pImage[3*Index1D + 1];
                c <<= 8;
                c |= pImage[3*Index1D + 2];
                c |= 0xFF000000;

                ( ( LONG* ) colorRGBX ) [y*width + x] = c;
            }
        }
    }


    return hr;
}
