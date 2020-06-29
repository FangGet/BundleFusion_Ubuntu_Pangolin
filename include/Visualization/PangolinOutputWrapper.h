#pragma once

#include <Output3DWrapper.h>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include <boost/thread.hpp>

#include <eigen3/Eigen/Core>

typedef Eigen::Matrix<float,3,1> Vec3f;
typedef Eigen::Matrix<unsigned int,int ( 3 ),int ( 1 ) > Vec3b;

namespace Visualization
{


struct translation {
    float x;
    float y;
    float z;
};

class PangolinOutputWrapper: public Output3DWrapper
{
public:
    PangolinOutputWrapper ( int width, int height );

    virtual ~PangolinOutputWrapper();

    void run();

    void close();

    virtual void publishSurface ( const MarchingCubesData* mesh );

    virtual void publishDepthMap ( float* data );

    virtual void publishColorMap ( uchar4* rgba );

    virtual void publishColorRayCastedMap ( float4* rgba );

    virtual void publishDepthRayCastedMap ( float* depth );

    virtual void noticeFinishFlag();

    virtual bool getPublishRGBFlag();

    virtual bool getPublishMeshFlag();

    virtual void publishAllTrajetory ( float* trajs, int size );

    virtual void publishCurrentCameraPose ( float* pose );


    virtual void join();

    virtual void reset();
private:

    void drawMesh();

    void drawCam ( float lineWidth, float* color,float sizeFactor );

    bool getCurrentOpenGLCameraMatrix ( pangolin::OpenGlMatrix& M );

    bool settings_followCamera;
    bool settings_show3D;
    bool settings_showLiveDepth;
    bool settings_showLiveColor;
    bool settings_showTraj;
    bool settings_showCamera;
    bool settings_resetButton;

    int width,height;
    bool needReset;
    void reset_internal();

    boost::thread runThread;
    bool running;

    bool ModelChanged;

    //images render;
    boost::mutex openImagesMutex;
    char* depthImg;
    char* colorImg;

    bool colorImgChanged,depthImgChanged;


    //3D model rendering
    boost::mutex model3DMutex;

    boost::mutex modelTrajMutex;

    MarchingCubesData* march_cube;
    // Create vertex and colour buffer objects and register them with CUDA
    pangolin::GlBufferCudaPtr* vertex_array_global;
    pangolin::GlBufferCudaPtr* indices_array_global;
    pangolin::GlBufferCudaPtr* colour_array_global;

    int mesh_save_idx = 0;

    std::vector<translation> trans;

    std::vector<float> currentPose;
};

}
