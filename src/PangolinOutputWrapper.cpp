#include <PangolinOutputWrapper.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <mLibCuda.h>
#include <BundleFusion.h>
#include <GlobalAppState.h>
#include <unistd.h>


extern "C" void launch_convert_kernel ( float3* march_cube,  float3* dVertexArray, uchar3* dColourArray, uint3* dIndicesArray,
                                        unsigned int size );

namespace Visualization
{

PangolinOutputWrapper::PangolinOutputWrapper ( int width, int height )
{
    this->width = width;
    this->height = height;
    running = true;

    colorImg = new char[width * height * 4];
    depthImg = new char[width * height * 3];
    memset ( colorImg, 128, sizeof ( char ) * width * height * 4 );
    memset ( depthImg, 128, sizeof ( char ) * width * height * 3 );

    march_cube = nullptr;
    vertex_array_global = nullptr;
    indices_array_global = nullptr;
    colour_array_global = nullptr;

    this->needReset = false;
    ModelChanged = false;
    runThread = boost::thread ( &PangolinOutputWrapper::run, this );
}

PangolinOutputWrapper::~PangolinOutputWrapper()
{

    close();

    //join();
}

void PangolinOutputWrapper::run()
{
    pangolin::CreateWindowAndBind ( "BundleFusion", 4 * width, 4 * height );
    glewInit();
    const int UI_WIDTH = 180;
    glEnable ( GL_DEPTH_TEST );

    //3D visualization
    pangolin::OpenGlRenderState visualizer ( pangolin::ProjectionMatrix ( width,height,400,400,width/2,height/2,0.1,1000 ),pangolin::ModelViewLookAt ( -0,-5,-10, 0,0,0, pangolin::AxisNegY ) );
    pangolin::View& displayer=pangolin::CreateDisplay()
                              .SetBounds ( 0.0,1.0,pangolin::Attach::Pix ( UI_WIDTH ),1.0,-width/ ( float ) height )
                              .SetHandler ( new pangolin::Handler3D ( visualizer ) );


    //2images
    pangolin::View& d_color=pangolin::Display ( "imgColor" ).SetAspect ( width/ ( float ) height );
    pangolin::GlTexture tex_color ( width,height,GL_RGBA,false,0,GL_RGBA,GL_UNSIGNED_BYTE );


    pangolin::View& images_displayer=pangolin::CreateDisplay()
                                     .SetBounds ( 0.0,0.3,pangolin::Attach::Pix ( UI_WIDTH ),1.0 )
                                     .SetLayout ( pangolin::LayoutEqual )
                                     .AddDisplay ( d_color );

    //paramter reconfigure gui
    pangolin::CreatePanel ( "ui" ).SetBounds ( 0.0,1.0,0.0,pangolin::Attach::Pix ( UI_WIDTH * 1.3 ) );

    pangolin::Var<bool> settings_followCamera ( "ui.followCam",false,true );
    pangolin::Var<bool> settings_show3D ( "ui.showMesh",true,true );
    pangolin::Var<bool> settings_showLiveColor ( "ui.showRGB",true,true );

    pangolin::Var<bool> settings_showCamera ( "ui.showCam",true,true );
    pangolin::Var<bool> settings_showAllTraj ( "ui.showTraj",true,true );

    pangolin::Var<bool> settings_saveMesh ( "ui.saveMesh",false,false );

    pangolin::Var<bool> settings_resetButton ( "ui.Reset",false,false );

    pangolin::Var<std::string> settings_GPUAllMemory ( "ui.GPU All","",0,0,false );
    pangolin::Var<std::string> settings_GPURestMemory ( "ui.GPU Left","",0,0,false );

    pangolin::Var<std::string> settings_FrameIdx ( "ui.Frame Count","",0,0,false );
    pangolin::Var<std::string> settings_GPUUsage ( "ui.Mesh GPU","0 MB",0,0,false );


    bool camera_follow=true;

    int count = 0;

    double start_gpu_memory = 0.0;

    pangolin::OpenGlMatrix M;
    M.SetIdentity();

    while ( !pangolin::ShouldQuit() &&running )
    {
        //clear entire screen
        glClearColor ( 0.0f, 0.0f, 0.0f,0.6f );
        glClear ( GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT );

        size_t mf, ma;
        cudaMemGetInfo ( &mf, &ma );

        settings_GPUAllMemory = std::to_string ( ( double ) ma / ( 1024*1024 ) ) + " MB";
        settings_GPURestMemory = std::to_string ( ( double ) mf / ( 1024*1024 ) ) + " MB";
        settings_FrameIdx = std::to_string ( count );
        if ( start_gpu_memory != 0 )
        {
            settings_GPUUsage = std::to_string ( double ( start_gpu_memory - mf ) / ( 1024*1024 ) ) + " MB";
            //settings_GPUAvgUsage = std::to_string ( double ( start_gpu_memory - mf ) / ( 1024*1024 ) / ( count + 1 ) ) + " MB";
        }

        displayer.Activate ( visualizer );
        if ( this->settings_show3D )
        {
            //Active efficiently by object

            boost::unique_lock<boost::mutex> lk3d ( model3DMutex );

            if ( currentPose.size() == 16 )
            {

                if ( getCurrentOpenGLCameraMatrix ( M ) )
                {

                    if ( this->settings_followCamera &&camera_follow )
                    {
                        visualizer.Follow ( M );
                    }
                    else if ( this->settings_followCamera && !camera_follow )
                    {
                        visualizer.SetModelViewMatrix ( pangolin::ModelViewLookAt ( 0,-0.7,-1.8,0,0,0,0.0,-1.0,0.0 ) );
                        visualizer.Follow ( M );
                        camera_follow=true;
                    }
                    else if ( !this->settings_followCamera && camera_follow )
                    {
                        camera_follow = false;
                    }
                }
            }

            if ( ModelChanged )
            {
                if ( march_cube != nullptr )
                {
                    uint* size = new uint ( 0 );
                    cutilSafeCall ( cudaMemcpy ( size, march_cube->d_numTriangles, sizeof ( uint ), cudaMemcpyDeviceToHost ) );

                    // debug log
                    std::cout<<"============================================="<<std::endl;
                    std::cout<<"== Triangle Size: " << *size << std::endl;
                    std::cout<<"============================================="<<std::endl;
                    int size_x64 = int ( *size );

                    //size_x64 = size_x64 - size_x64 % 64;


                    if ( vertex_array_global == nullptr )
                    {
                        vertex_array_global = new pangolin::GlBufferCudaPtr (
                            pangolin::GlArrayBuffer, size_x64 * 3, GL_FLOAT, 3,
                            cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW
                        );
                    }
                    else
                    {
                        vertex_array_global->Free();
                        vertex_array_global->Reinitialise ( pangolin::GlArrayBuffer, size_x64 * 3, GL_FLOAT, 3,
                                                            cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW );
                    }

                    if ( indices_array_global == nullptr )
                    {
                        indices_array_global = new pangolin::GlBufferCudaPtr (
                            pangolin::GlElementArrayBuffer, size_x64, GL_UNSIGNED_INT, 3,
                            cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW
                        );
                    }
                    else
                    {
                        indices_array_global->Free();
                        indices_array_global->Reinitialise ( pangolin::GlElementArrayBuffer, size_x64, GL_UNSIGNED_INT, 3,
                                                             cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW );
                    }

                    if ( colour_array_global == nullptr )
                    {
                        colour_array_global = new pangolin::GlBufferCudaPtr (
                            pangolin::GlArrayBuffer, size_x64 * 3, GL_UNSIGNED_BYTE, 3,
                            cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW
                        );
                    }
                    else
                    {
                        colour_array_global->Free();
                        colour_array_global->Reinitialise ( pangolin::GlArrayBuffer, size_x64 * 3, GL_UNSIGNED_BYTE, 3,
                                                            cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW );
                    }


                    pangolin::CudaScopedMappedPtr var ( *vertex_array_global );
                    pangolin::CudaScopedMappedPtr iar ( *indices_array_global );
                    pangolin::CudaScopedMappedPtr car ( *colour_array_global );

                    launch_convert_kernel ( ( float3 * ) ( march_cube->d_triangles ), ( float3* ) *var, ( uchar3* ) *car, ( uint3* ) *iar, size_x64 );


                    march_cube = nullptr;
                }
//                 }
            }

            //drawMesh();
            lk3d.unlock();
        }

        {
            boost::unique_lock<boost::mutex> lk3d ( model3DMutex );
            if ( colour_array_global )
            {
                colour_array_global->Bind();
                glColorPointer ( colour_array_global->count_per_element, colour_array_global->datatype, 0, 0 );
                glEnableClientState ( GL_COLOR_ARRAY );
            }
            if ( vertex_array_global )
            {
                vertex_array_global->Bind();
                glVertexPointer ( vertex_array_global->count_per_element, vertex_array_global->datatype, 0, 0 );
                glEnableClientState ( GL_VERTEX_ARRAY );
            }

            if ( indices_array_global )
            {
                indices_array_global->Bind();
                glDrawElements ( GL_TRIANGLES, indices_array_global->num_elements*indices_array_global->count_per_element, indices_array_global->datatype, 0 );
                indices_array_global->Unbind();
            }


            if ( indices_array_global )
            {
                glDisableClientState ( GL_VERTEX_ARRAY );
                indices_array_global->Unbind();
            }
            if ( vertex_array_global )
            {
                glDisableClientState ( GL_VERTEX_ARRAY );
                vertex_array_global->Unbind();
            }
            if ( colour_array_global )
            {
                glDisableClientState ( GL_COLOR_ARRAY );
                colour_array_global->Unbind();
            }
            lk3d.unlock();
        }

        if ( this->settings_showTraj )
        {
            boost::unique_lock<boost::mutex> lk ( modelTrajMutex );

            float color_blue[3] = {1, 1, 0};
            glColor3f ( 0,1,0 );

            glLineWidth ( 2 );
            glBegin ( GL_LINE_STRIP );

            for ( size_t i = 0; i < trans.size(); ++i )
            {
                glVertex3f ( trans[i].x,trans[i].y, trans[i].z );
            }

            glEnd();

            if ( trans.size() > 10 )
            {
                glLineWidth ( 5 );
                glColor3f ( 1,0,0 );
                glBegin ( GL_LINE_STRIP );

                for ( size_t i = trans.size()- 10 ; i < trans.size(); ++i )
                {
                    glVertex3f ( trans[i].x,trans[i].y, trans[i].z );
                }

                glEnd();
            }

        }

        if ( this->settings_showCamera )
        {
            boost::unique_lock<boost::mutex> lk ( modelTrajMutex );
            if ( currentPose.size() == 16 )
            {
                this->drawCam ( 2., 0, 0.2 );
            }
        }

        {
            openImagesMutex.lock();
            if ( colorImgChanged )
            {

                if ( count == 0 )
                {
                    size_t mef, mea;
                    cudaMemGetInfo ( &mef, &mea );
                    start_gpu_memory = mef;
                }
                count++;

                tex_color.Upload ( colorImg,GL_RGBA,GL_UNSIGNED_BYTE );
            }

            depthImgChanged=colorImgChanged=false;
            ModelChanged = false;

            openImagesMutex.unlock();
        }

        if ( this->settings_showLiveColor )
        {
            d_color.Activate();
            glColor4f ( 1.0f,1.0f,1.0f,1.0f );
            tex_color.RenderToViewportFlipY();
        }

        // update parameters
        this->settings_followCamera = settings_followCamera.Get();
        this->settings_show3D=settings_show3D.Get();
        this->settings_showLiveColor=settings_showLiveColor.Get();
        this->settings_showCamera = settings_showCamera.Get();
        this->settings_showTraj = settings_showAllTraj.Get();

        setPublishRGBFlag ( this->settings_showLiveColor );

        setPublishMeshFlag ( this->settings_show3D );

        if ( settings_resetButton.Get() )
        {
            printf ( "Pangolin Viewer Reset ...\n" );
            settings_resetButton.Reset();
        }

        if ( settings_saveMesh.Get() )
        {
            std::cout<<"Pangolin Viewer save mesh into file" << std::endl;

            struct timeval t;
            gettimeofday ( &t,nullptr );

            saveMeshIntoFile ( GlobalAppState::get().s_generateMeshDir + "scan_" + std::to_string ( t.tv_sec ) + ".ply", true );

            settings_saveMesh.Reset();
        }

        if ( needReset )
        {
            reset_internal();
            camera_follow=true;
        }


        pangolin::FinishFrame();

    }

    if ( depthImg != nullptr )
    {
        delete depthImg;
        depthImg = nullptr;
    }


    if ( colorImg != nullptr )
    {
        delete colorImg;
        colorImg = nullptr;
    }


    if ( vertex_array_global != nullptr )
    {
        vertex_array_global->Free();
        vertex_array_global = nullptr;
    }

    if ( indices_array_global != nullptr )
    {
        indices_array_global->Free();
        indices_array_global = nullptr;
    }

    if ( colour_array_global != nullptr )
    {
        colour_array_global->Free();
        colour_array_global = nullptr;
    }

    printf ( "Pangolin is finished ... \n" );
    exit ( 1 );
}

void PangolinOutputWrapper::noticeFinishFlag()
{
    close();
    // join();
}

void PangolinOutputWrapper::close()
{
    running = false;
}

void PangolinOutputWrapper::join()
{
    runThread.join();
    printf ( "Joined Pangolin thread... \n" );
}

void PangolinOutputWrapper::reset()
{
    needReset = true;
}

void PangolinOutputWrapper::reset_internal()
{

    openImagesMutex.lock();
    delete colorImg;
    colorImg=0;
    delete depthImg;
    depthImg=0;
    colorImgChanged= depthImgChanged=false;
    openImagesMutex.unlock();

    needReset = false;
}

bool PangolinOutputWrapper::getPublishRGBFlag()
{
    return this->settings_showLiveColor;
}

bool PangolinOutputWrapper::getPublishMeshFlag()
{
    return this->settings_show3D;
}

void PangolinOutputWrapper::publishSurface ( const  MarchingCubesData* m_cube )
{
    if ( running )
    {
        march_cube = m_cube;
        ModelChanged = true;
    }

}

void PangolinOutputWrapper::publishDepthMap ( float* data )
{
    if ( running )
    {
        boost::unique_lock<boost::mutex> lk ( openImagesMutex );
        memcpy ( depthImg, data, width*height*3 );
        depthImgChanged = true;
    }
}

void PangolinOutputWrapper::publishColorMap ( uchar4* rgba )
{
    if ( running )
    {
        boost::unique_lock<boost::mutex> lk ( openImagesMutex );
        const unsigned char * d_color_uchar = reinterpret_cast<const unsigned char*> ( rgba );
        memcpy ( colorImg, rgba, width*height*4 );
        colorImgChanged = true;
    }
}

void PangolinOutputWrapper::publishColorRayCastedMap ( float4* rgba )
{
    //TODO
}

void PangolinOutputWrapper::publishDepthRayCastedMap ( float* depth )
{
    //TODO
}

void PangolinOutputWrapper::publishAllTrajetory ( float* trajs, int size )
{
    if ( running )
    {
        boost::unique_lock<boost::mutex> lk ( modelTrajMutex );

        trans.clear();
        trans.resize ( size );

        for ( size_t i = 0; i < size; ++i )
        {
            trans[i].x = trajs[3*i + 0];
            trans[i].y = trajs[3*i + 1];
            trans[i].z = trajs[3*i + 2];
        }
    }
}
void PangolinOutputWrapper::publishCurrentCameraPose ( float* pose )
{
    if ( running )
    {
        boost::unique_lock<boost::mutex> lk ( modelTrajMutex );

        if ( currentPose.empty() )
        {
            currentPose.resize ( 16 );
        }

        for ( int i = 0; i < 16; ++i )
        {
            currentPose[i] = pose[i];
        }
    }
}


void PangolinOutputWrapper::drawMesh()
{
}

void PangolinOutputWrapper::drawCam ( float lineWidth, float* color,float sizeFactor )
{
    if ( lineWidth == 0 )
    {
        return;
    }

    float sz=sizeFactor;
    glPushMatrix();

    //std::cout<<"/*****************************************/" << std::endl;
    //for(int i = 0; i < 16; ++i){
    //  std::cout<<currentPose[i] << " ";
    //
    //}
    //std::cout<<std::endl;
    //std::cout<<"/==========================================/"<<std::endl;

    glMultMatrixf ( ( GLfloat* ) ( currentPose.data() ) );

    if ( color == 0 )
    {
        glColor3f ( 1,0,0 );
    }
    else
    {
        glColor3f ( color[0],color[1],color[2] );
    }

    glLineWidth ( lineWidth );
    glBegin ( GL_LINES );

/////////////////////////////////////
///orb-like
/////////////////////////////////////
    const float& w=sz;
    const float h=w*0.75;
    const float z=w*0.6;

    glVertex3f ( 0,0,0 );
    glVertex3f ( w,h,z );

    glVertex3f ( 0,0,0 );
    glVertex3f ( w,-h,z );

    glVertex3f ( 0,0,0 );
    glVertex3f ( -w,-h,z );

    glVertex3f ( 0,0,0 );
    glVertex3f ( -w,h,z );

    glVertex3f ( w,h,z );
    glVertex3f ( w,-h,z );

    glVertex3f ( -w,h,z );
    glVertex3f ( -w,-h,z );

    glVertex3f ( -w,h,z );
    glVertex3f ( w,h,z );

    glVertex3f ( -w,-h,z );
    glVertex3f ( w,-h,z );

    glEnd();
    glPopMatrix();
}

bool PangolinOutputWrapper::getCurrentOpenGLCameraMatrix ( pangolin::OpenGlMatrix& M )
{
    if ( currentPose.size() != 16 )
    {
        return false;
    }
//   int* idxs = new int[16]{0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15};
    for ( size_t i = 0; i < 16; ++i )
    {
        M.m[i] = currentPose[i];
        //std::cout<<M.m[i] << " ";
    }
    //std::cout<<std::endl;
    return true;
}







}





