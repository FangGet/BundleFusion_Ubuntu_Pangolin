#include <BundleFusion.h>
#include <PrimeSenseSensor.h>
#include <GlobalBundlingState.h>
#include <TimingLog.h>

#include <SiftGPU/MatrixConversion.h>
#include <SiftGPU/CUDATimer.h>
#include <SiftGPU/SIFTMatchFilter.h>
#include <CUDAImageManager.h>

#include <ConditionManager.h>
#include <DualGPU.h>
#include <OnlineBundler.h>

#include <RGBDSensor.h>
#include <SensorDataReader.h>
#include <TrajectoryManager.h>


#include <GlobalAppState.h>

#include <DepthSensing/TimingLogDepthSensing.h>
#include <DepthSensing/Util.h>
#include <DepthSensing/CUDASceneRepHashSDF.h>
#include <DepthSensing/CUDARayCastSDF.h>
#include <DepthSensing/CUDAMarchingCubesHashSDF.h>
#include <DepthSensing/CUDAHistogramHashSDF.h>
#include <DepthSensing/CUDASceneRepChunkGrid.h>
#include <CUDAImageManager.h>

#include <iomanip>
#include <fstream>
#include <unistd.h>

#ifdef WITH_VISUALIZATION
#include <Output3DWrapper.h>
#include <PangolinOutputWrapper.h>
#endif

// Variables
RGBDSensor* g_RGBDSensor = nullptr;
CUDAImageManager* g_imageManager = nullptr;
OnlineBundler* g_bundler = nullptr;
#ifdef WITH_VISUALIZATION
Visualization::Output3DWrapper * wrapper = nullptr;
#endif

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CUDASceneRepHashSDF*		g_sceneRep = NULL;
CUDARayCastSDF*				g_rayCast = NULL;
CUDAMarchingCubesHashSDF*	g_marchingCubesHashSDF = NULL;
CUDAHistrogramHashSDF*		g_historgram = NULL;
CUDASceneRepChunkGrid*		g_chunkGrid = NULL;

DepthCameraParams			g_depthCameraParams;
mat4f						g_lastRigidTransform = mat4f::identity();

//managed externally
mat4f g_transformWorld = mat4f::identity();

int surface_read_count = 0;
bool publish_rgb = true;
bool publish_depth = false;
bool publish_mesh = true;

std::thread* bundlingThread;


// Functions
/**
 * debug function
 * get rgbdSensor, current default select PrimeSenseSensor
 * */
RGBDSensor* getRGBDSensor();

void bundlingOptimization();
void bundlingOptimizationThreadFunc();
void bundlingThreadFunc();

void integrate ( const DepthCameraData& depthCameraData, const mat4f& transformation );
void deIntegrate ( const DepthCameraData& depthCameraData, const mat4f& transformation );
void reintegrate();


void StopScanningAndExtractIsoSurfaceMC ( const std::string& filename = "./scans/scan.ply", bool overwriteExistingFile = false );
void StopScanningAndExit ( bool aborted = false );

void ResetDepthSensing();

bool CreateDevice();
extern "C" void convertColorFloat4ToUCHAR4 ( uchar4* d_output, float4* d_input, unsigned int width, unsigned int height );

/*************************BundleFusion SDK Interface ********************/
bool initBundleFusion ( std::string app_config, std::string bundle_config )
{
    try
    {
        if ( app_config.empty() || bundle_config.empty() )
        {
            std::cerr<<"app/bundle configure file is empty." <<std::endl;
            return false;
        }

        //Read the global app state
        ParameterFile parameterFileGlobalApp ( app_config );

        GlobalAppState::getInstance().readMembers ( parameterFileGlobalApp );

        //Read the global camera tracking state
        ParameterFile parameterFileGlobalBundling ( bundle_config );
        GlobalBundlingState::getInstance().readMembers ( parameterFileGlobalBundling );

        DualGPU& dualGPU = DualGPU::get();	//needs to be called to initialize devices
        dualGPU.setDevice ( DualGPU::DEVICE_RECONSTRUCTION );	//main gpu
        ConditionManager::init();

        g_RGBDSensor = getRGBDSensor();

        //init the input RGBD sensor
        if ( g_RGBDSensor == NULL )
        {
            std::cerr<<"No RGBD sensor specified."<<std::endl;
            return false;
        }
        g_RGBDSensor->createFirstConnected();


        g_imageManager = new CUDAImageManager ( GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
                                                GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT, g_RGBDSensor, false );
#ifdef RUN_MULTITHREADED
        bundlingThread  = new std::thread ( bundlingThreadFunc );
        //waiting until bundler is initialized
        while ( !g_bundler )	usleep ( 0 );
#else
        g_bundler = new OnlineBundler ( g_RGBDSensor, g_imageManager );
#endif

        dualGPU.setDevice ( DualGPU::DEVICE_RECONSTRUCTION );	//main gpu

#ifdef WITH_VISUALIZATION
        wrapper = new Visualization::PangolinOutputWrapper ( GlobalAppState::get().s_integrationWidth,GlobalAppState::get().s_integrationHeight );
#endif

        if ( !CreateDevice() )
        {
            std::cerr<<"Create Device failed. " << std::endl;
            return false;
        }



        if ( GlobalAppState::get().s_generateVideo ) g_transformWorld = GlobalAppState::get().s_topVideoTransformWorld;
    }
    catch ( const std::exception& e )
    {
        //MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
        std::cerr<< ( "Exception caught" ) << std::endl;
        return false;
    }
    catch ( ... )
    {
        //MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
        std::cerr<< ( "UNKNOWN EXCEPTION" ) << std::endl;;
        return false;
    }



    return true;
}

bool processInputRGBDFrame ( cv::Mat& rgb, cv::Mat& depth )
{
    //printf("START FrameRender\n");
    if ( ConditionManager::shouldExit() )
    {
        //StopScanningAndExit ( true );

        return false;
    }

    // Read Input
    ///////////////////////////////////////
#ifdef RUN_MULTITHREADED
    ConditionManager::lockImageManagerFrameReady ( ConditionManager::Recon );
    while ( g_imageManager->hasBundlingFrameRdy() ) //wait until bundling is done with previous frame
    {
        ConditionManager::waitImageManagerFrameReady ( ConditionManager::Recon );
    }
    bool bGotDepth = g_imageManager->process ( rgb, depth );
    if ( bGotDepth )
    {
        g_imageManager->setBundlingFrameRdy();					//ready for bundling thread
        ConditionManager::unlockAndNotifyImageManagerFrameReady ( ConditionManager::Recon );
    }
    if ( !g_RGBDSensor->isReceivingFrames() ) //sequence is done
    {
        if ( bGotDepth )
        {
            std::cerr << ( "ERROR bGotDepth = true but sequence is done" ) << std::endl;
            return false;
        }

        g_imageManager->setBundlingFrameRdy();				// let bundling still optimize after scanning done
        ConditionManager::unlockAndNotifyImageManagerFrameReady ( ConditionManager::Recon );
    }
#else
    bool bGotDepth = g_imageManager->process ( rgb, depth );
    g_depthSensingBundler->processInput();
#endif

    ///////////////////////////////////////
    // Fix old frames
    ///////////////////////////////////////
    //printf("start reintegrate\n");
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    reintegrate();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    //printf("end reintegrate\n");
#ifdef RUN_MULTITHREADED
    //wait until the bundling thread is done with: sift extraction, sift matching, and key point filtering
    ConditionManager::lockBundlerProcessedInput ( ConditionManager::Recon );
    while ( !g_bundler->hasProcssedInputFrame() ) ConditionManager::waitBundlerProcessedInput ( ConditionManager::Recon );

    if ( !g_RGBDSensor->isReceivingFrames() ) // let bundling still optimize after scanning done
    {
        g_bundler->confirmProcessedInputFrame();
        ConditionManager::unlockAndNotifyBundlerProcessedInput ( ConditionManager::Recon );
    }
#endif

    ///////////////////////////////////////
    // Reconstruction of current frame
    ///////////////////////////////////////
    bool validTransform = true;
    bool bGlobalTrackingLost = false;
    if ( bGotDepth )
    {
        mat4f transformation = mat4f::zero();
        unsigned int frameIdx;
        validTransform = g_bundler->getCurrentIntegrationFrame ( transformation, frameIdx, bGlobalTrackingLost );
#ifdef RUN_MULTITHREADED
        //allow bundler to process new frame
        g_bundler->confirmProcessedInputFrame();
        ConditionManager::unlockAndNotifyBundlerProcessedInput ( ConditionManager::Recon );
#endif

        if ( GlobalAppState::get().s_binaryDumpSensorUseTrajectory && GlobalAppState::get().s_sensorIdx == 3 )
        {
            //overwrite transform and use given trajectory in this case
            transformation = g_RGBDSensor->getRigidTransform();
            validTransform = true;
        }

        if ( GlobalAppState::getInstance().s_recordData )
        {
            g_RGBDSensor->recordFrame();

        }

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        if ( validTransform && GlobalAppState::get().s_reconstructionEnabled )
        {
            DepthCameraData depthCameraData ( g_imageManager->getIntegrateFrame ( frameIdx ).getDepthFrameGPU(), g_imageManager->getIntegrateFrame ( frameIdx ).getColorFrameGPU() );
            integrate ( depthCameraData, transformation );
            g_bundler->getTrajectoryManager()->addFrame ( TrajectoryManager::TrajectoryFrame::Integrated, transformation, g_imageManager->getCurrFrameNumber() );
        }
        else
        {
            g_bundler->getTrajectoryManager()->addFrame ( TrajectoryManager::TrajectoryFrame::NotIntegrated_NoTransform, mat4f::zero ( -std::numeric_limits<float>::infinity() ), g_imageManager->getCurrFrameNumber() );
        }
        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> ( t2 - t1 +t4 - t3 );
        std::cout << "depthSensing time cost = " << time_used.count() << " seconds." << std::endl;

        std::chrono::duration<double> time_total = std::chrono::duration_cast<std::chrono::duration<double>> ( t4 - t1 );
        std::cout << "total time cost = " << time_total.count() << " seconds." << std::endl;

        if ( validTransform )
        {
            g_lastRigidTransform = transformation;
        }
    }


    ///////////////////////////////////////////
    ////// Bundling Optimization
    ///////////////////////////////////////////
#ifndef RUN_MULTITHREADED
    g_depthSensingBundler->process ( GlobalBundlingState::get().s_numLocalNonLinIterations, GlobalBundlingState::get().s_numLocalLinIterations,
                                     GlobalBundlingState::get().s_numGlobalNonLinIterations, GlobalBundlingState::get().s_numGlobalLinIterations );
#endif

#ifdef WITH_VISUALIZATION
    if ( wrapper != nullptr )
    {
        // raw color image
        const uchar4* d_color = g_imageManager->getLastIntegrateFrame().getColorFrameCPU();
        // raw depth image
        const float* d_depth = g_imageManager->getLastIntegrateFrame().getDepthFrameCPU();
        const float minDepth = GlobalAppState::get().s_sensorDepthMin;
        const float maxDepth = GlobalAppState::get().s_sensorDepthMax;

        // publish these data
        if ( publish_rgb )
        {
            wrapper->publishColorMap ( d_color );
        }

        if ( publish_depth )
        {
            wrapper->publishDepthMap ( d_depth );
        }

        if ( publish_mesh )
        {
            // surface get
            MarchingCubesData* march_cube = nullptr;

            if ( surface_read_count == 1 )
            {
                surface_read_count = 0;

                if ( GlobalAppState::get().s_sensorIdx == 7 ) //! hack for structure sensor
                {
                    std::cout << "[marching cubes] stopped receiving frames from structure sensor" << std::endl;
                    g_RGBDSensor->stopReceivingFrames();
                }

                Timer t;

                march_cube = g_marchingCubesHashSDF->extractIsoSurfaceGPUNoChrunk ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );

                std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;
            }
            else
            {
                surface_read_count++;
            }

            if ( march_cube != nullptr )
            {
                wrapper->publishSurface ( march_cube );
            }

            std::vector<mat4f> trajectory;
            g_bundler->getTrajectoryManager()->getOptimizedTransforms ( trajectory );
            float* trajs_float = new float[3 * trajectory.size()];
            float* pose = new float[16];
            for ( size_t i = 0; i < trajectory.size(); ++i )
            {
                trajs_float[3*i + 0] = trajectory[i][3];
                trajs_float[3*i + 1] = trajectory[i][7];
                trajs_float[3*i + 2] = trajectory[i][11];
                if ( i == trajectory.size()-1 )
                {
                    mat4f& last_pose = trajectory[i];
                    last_pose.transpose();
                    for ( size_t j = 0; j < 16; ++j )
                        pose[j] = last_pose[j];
                }

            }

            wrapper->publishAllTrajetory ( trajs_float, trajectory.size() );


            wrapper->publishCurrentCameraPose ( pose );
//
            delete pose;

            delete trajs_float;

        }

    }
#endif

    return true;

}

void setPublishRGBFlag ( bool publish_flag )
{
    publish_rgb = publish_flag;
}

void setPublishMeshFlag ( bool publish_flag )
{
    publish_mesh = publish_flag;
}

bool saveMeshIntoFile ( const std::string& filename, bool overwriteExistingFile /*= false*/ )
{
    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
    if ( GlobalAppState::get().s_sensorIdx == 7 ) //! hack for structure sensor
    {
        std::cout << "[marching cubes] stopped receiving frames from structure sensor" << std::endl;
        g_RGBDSensor->stopReceivingFrames();
    }
    std::cout << "running marching cubes...1" << std::endl;

    Timer t;


    g_marchingCubesHashSDF->clearMeshBuffer();
    if ( !GlobalAppState::get().s_streamingEnabled )
    {
        //g_chunkGrid->stopMultiThreading();
        //g_chunkGrid->streamInToGPUAll();
        g_marchingCubesHashSDF->extractIsoSurface ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );
        //g_chunkGrid->startMultiThreading();
    }
    else
    {
        vec4f posWorld = vec4f ( g_lastRigidTransform*GlobalAppState::get().s_streamingPos, 1.0f ); // trans lags one frame
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );
        g_marchingCubesHashSDF->extractIsoSurface ( *g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius );
    }

    const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
    g_marchingCubesHashSDF->saveMesh ( filename, &rigidTransform, overwriteExistingFile );

    std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

    return true;

    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
}

bool deinitBundleFusion()
{

    SAFE_DELETE ( g_sceneRep );
    SAFE_DELETE ( g_rayCast );
    SAFE_DELETE ( g_marchingCubesHashSDF );
    SAFE_DELETE ( g_historgram );
    SAFE_DELETE ( g_chunkGrid );

#ifdef RUN_MULTITHREADED
    g_bundler->exitBundlingThread();

    g_imageManager->setBundlingFrameRdy();			//release all bundling locks
    g_bundler->confirmProcessedInputFrame();		//release all bundling locks
    ConditionManager::release ( ConditionManager::Recon ); // release bundling locks

    if ( bundlingThread->joinable() )
        bundlingThread->join();	//wait for the bundling thread to return;
#endif
    SAFE_DELETE ( g_bundler );
    SAFE_DELETE ( g_imageManager );

    //ConditionManager::DEBUGRELEASE();

    //this is a bit of a hack due to a bug in std::thread (a static object cannot join if the main thread exists)
    auto* s = getRGBDSensor();
    SAFE_DELETE ( s );

#ifdef WITH_VISUALIZATION
    if ( wrapper != nullptr )
    {
        wrapper->noticeFinishFlag();
    }
#endif

    return true;

}

/************************************************************************/

RGBDSensor* getRGBDSensor()
{
    static RGBDSensor* g_sensor = NULL;
    if ( g_sensor != NULL )	return g_sensor;

    if ( GlobalAppState::get().s_sensorIdx == 1 )
    {
        //static PrimeSenseSensor s_primeSense;
        //return &s_primeSense;
        g_sensor = new PrimeSenseSensor;
        return g_sensor;
    }
    throw MLIB_EXCEPTION ( "unkown sensor id " + std::to_string ( GlobalAppState::get().s_sensorIdx ) );

    return NULL;
}

void bundlingOptimization()
{
    g_bundler->process ( GlobalBundlingState::get().s_numLocalNonLinIterations, GlobalBundlingState::get().s_numLocalLinIterations,
                         GlobalBundlingState::get().s_numGlobalNonLinIterations, GlobalBundlingState::get().s_numGlobalLinIterations );
    //g_bundler->resetDEBUG(false, false); // for no opt
}

void bundlingOptimizationThreadFunc()
{

    DualGPU::get().setDevice ( DualGPU::DEVICE_BUNDLING );

    bundlingOptimization();
}

void bundlingThreadFunc()
{
    assert ( g_RGBDSensor && g_imageManager );
    DualGPU::get().setDevice ( DualGPU::DEVICE_BUNDLING );
    g_bundler = new OnlineBundler ( g_RGBDSensor, g_imageManager );

    std::thread tOpt;
    int submapSize = GlobalBundlingState::get().s_submapSize;
    while ( 1 )
    {
        // opt
        if ( g_RGBDSensor->isReceivingFrames() )
        {
            if ( g_bundler->getCurrProcessedFrame() % submapSize == 0 ) // stop solve
            {
                if ( tOpt.joinable() )
                {
                    tOpt.join();
                }
            }
            if ( g_bundler->getCurrProcessedFrame() % submapSize == 1 ) // start solve
            {
                MLIB_ASSERT ( !tOpt.joinable() );
                tOpt = std::thread ( bundlingOptimizationThreadFunc );
            }
        }
        else   // stop then start solve
        {
            if ( tOpt.joinable() )
            {
                tOpt.join();
            }
            tOpt = std::thread ( bundlingOptimizationThreadFunc );
        }
        //wait for a new input frame (LOCK IMAGE MANAGER)
        ConditionManager::lockImageManagerFrameReady ( ConditionManager::Bundling );
        while ( !g_imageManager->hasBundlingFrameRdy() )
        {
            ConditionManager::waitImageManagerFrameReady ( ConditionManager::Bundling );
        }
        {
            ConditionManager::lockBundlerProcessedInput ( ConditionManager::Bundling );
            while ( g_bundler->hasProcssedInputFrame() ) //wait until depth sensing has confirmed the last one (WAITING THAT DEPTH SENSING RELEASES ITS LOCK)
            {
                ConditionManager::waitBundlerProcessedInput ( ConditionManager::Bundling );
            }
            {
                if ( g_bundler->getExitBundlingThread() )
                {
                    if ( tOpt.joinable() )
                    {
                        tOpt.join();
                    }
                    ConditionManager::release ( ConditionManager::Bundling );
                    break;
                }
                g_bundler->processInput();						//perform sift and whatever
            }
            g_bundler->setProcessedInputFrame();			//let depth sensing know we have a frame (UNLOCK BUNDLING)
            ConditionManager::unlockAndNotifyBundlerProcessedInput ( ConditionManager::Bundling );
        }
        g_imageManager->confirmRdyBundlingFrame();		//here it's processing with a new input frame  (GIVE DEPTH SENSING THE POSSIBLITY TO LOCK IF IT WANTS)
        ConditionManager::unlockAndNotifyImageManagerFrameReady ( ConditionManager::Bundling );

        if ( g_bundler->getExitBundlingThread() )
        {
            ConditionManager::release ( ConditionManager::Bundling );
            break;
        }
    }
}

bool CreateDevice()
{

    g_sceneRep = new CUDASceneRepHashSDF ( CUDASceneRepHashSDF::parametersFromGlobalAppState ( GlobalAppState::get() ) );
    //g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_imageManager->getColorIntrinsics(), g_CudaImageManager->getColorIntrinsicsInv()));
    g_rayCast = new CUDARayCastSDF ( CUDARayCastSDF::parametersFromGlobalAppState ( GlobalAppState::get(), g_imageManager->getDepthIntrinsics(), g_imageManager->getDepthIntrinsicsInv() ) );

    g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF ( CUDAMarchingCubesHashSDF::parametersFromGlobalAppState ( GlobalAppState::get() ) );
    g_historgram = new CUDAHistrogramHashSDF ( g_sceneRep->getHashParams() );

    if ( GlobalAppState::get().s_streamingEnabled )
    {
        g_chunkGrid = new CUDASceneRepChunkGrid ( g_sceneRep,
                GlobalAppState::get().s_streamingVoxelExtents,
                GlobalAppState::get().s_streamingGridDimensions,
                GlobalAppState::get().s_streamingMinGridPos,
                GlobalAppState::get().s_streamingInitialChunkListSize,
                GlobalAppState::get().s_streamingEnabled,
                GlobalAppState::get().s_streamingOutParts );
    }
    if ( !GlobalAppState::get().s_reconstructionEnabled )
    {
        GlobalAppState::get().s_RenderMode = 2;
    }

    g_depthCameraParams.fx = g_imageManager->getDepthIntrinsics() ( 0, 0 ); //TODO check intrinsics
    g_depthCameraParams.fy = g_imageManager->getDepthIntrinsics() ( 1, 1 );

    g_depthCameraParams.mx = g_imageManager->getDepthIntrinsics() ( 0, 2 );
    g_depthCameraParams.my = g_imageManager->getDepthIntrinsics() ( 1, 2 );
    g_depthCameraParams.m_sensorDepthWorldMin = GlobalAppState::get().s_renderDepthMin;
    g_depthCameraParams.m_sensorDepthWorldMax = GlobalAppState::get().s_renderDepthMax;
    g_depthCameraParams.m_imageWidth = g_imageManager->getIntegrationWidth();
    g_depthCameraParams.m_imageHeight = g_imageManager->getIntegrationHeight();
    std::cout<<g_depthCameraParams.fx << "," << g_depthCameraParams.fy << "," <<g_depthCameraParams.mx << "," <<g_depthCameraParams.my << "," <<g_depthCameraParams.m_sensorDepthWorldMin << "," <<g_depthCameraParams.m_sensorDepthWorldMax << "," <<g_depthCameraParams.m_imageWidth << "," <<g_depthCameraParams.m_imageHeight << std::endl;
    DepthCameraData::updateParams ( g_depthCameraParams );

    //std::vector<DXGI_FORMAT> rtfFormat;
    //rtfFormat.push_back(DXGI_FORMAT_R8G8B8A8_UNORM); // _SRGB
    //V_RETURN(g_RenderToFileTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_rayCastWidth, GlobalAppState::get().s_rayCastHeight, rtfFormat));

    //g_CudaImageManager->OnD3D11CreateDevice(pd3dDevice);

    if ( GlobalAppState::get().s_sensorIdx == 7 ) // structure sensor
    {
        g_RGBDSensor->startReceivingFrames();
    }
    return true;
}

void integrate ( const DepthCameraData& depthCameraData, const mat4f& transformation )
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        vec4f posWorld = transformation*vec4f ( GlobalAppState::getInstance().s_streamingPos, 1.0f ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }

    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid ) d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->integrate ( g_transformWorld * transformation, depthCameraData, g_depthCameraParams, d_bitMask );
    }
    //else {
    //	//compactification is required for the ray cast splatting
    //	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
    //}
}

void deIntegrate ( const DepthCameraData& depthCameraData, const mat4f& transformation )
{
    if ( GlobalAppState::get().s_streamingEnabled )
    {
        vec4f posWorld = transformation*vec4f ( GlobalAppState::getInstance().s_streamingPos, 1.0f ); // trans laggs one frame *trans
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );

        g_chunkGrid->streamOutToCPUPass0GPU ( p, GlobalAppState::get().s_streamingRadius, true, true );
        g_chunkGrid->streamInToGPUPass1GPU ( true );
    }

    if ( GlobalAppState::get().s_integrationEnabled )
    {
        unsigned int* d_bitMask = NULL;
        if ( g_chunkGrid ) d_bitMask = g_chunkGrid->getBitMaskGPU();
        g_sceneRep->deIntegrate ( g_transformWorld * transformation, depthCameraData, g_depthCameraParams, d_bitMask );
    }
    //else {
    //	//compactification is required for the ray cast splatting
    //	g_sceneRep->setLastRigidTransformAndCompactify(transformation);	//TODO check this
    //}
}



void reintegrate()
{
    const unsigned int maxPerFrameFixes = GlobalAppState::get().s_maxFrameFixes;
    TrajectoryManager* tm = g_bundler->getTrajectoryManager();
    //std::cout << "reintegrate():" << tm->getNumActiveOperations() << " : " << tm->getNumOptimizedFrames() << std::endl;

    if ( tm->getNumActiveOperations() < maxPerFrameFixes )
    {
        tm->generateUpdateLists();
        //if (GlobalBundlingState::get().s_verbose) {
        //	if (tm->getNumActiveOperations() == 0)
        //		std::cout << __FUNCTION__ << " :  no more work (everything is reintegrated)" << std::endl;
        //}
    }

    for ( unsigned int fixes = 0; fixes < maxPerFrameFixes; fixes++ )
    {

        mat4f newTransform = mat4f::zero();
        mat4f oldTransform = mat4f::zero();
        unsigned int frameIdx = ( unsigned int )-1;

        if ( tm->getTopFromDeIntegrateList ( oldTransform, frameIdx ) )
        {
            auto& f = g_imageManager->getIntegrateFrame ( frameIdx );
            DepthCameraData depthCameraData ( f.getDepthFrameGPU(), f.getColorFrameGPU() );
            MLIB_ASSERT ( !isnan ( oldTransform[0] ) && oldTransform[0] != -std::numeric_limits<float>::infinity() );
            deIntegrate ( depthCameraData, oldTransform );
            continue;
        }
        else if ( tm->getTopFromIntegrateList ( newTransform, frameIdx ) )
        {
            auto& f = g_imageManager->getIntegrateFrame ( frameIdx );
            DepthCameraData depthCameraData ( f.getDepthFrameGPU(), f.getColorFrameGPU() );
            MLIB_ASSERT ( !isnan ( newTransform[0] ) && newTransform[0] != -std::numeric_limits<float>::infinity() );
            integrate ( depthCameraData, newTransform );
            tm->confirmIntegration ( frameIdx );
            continue;
        }
        else if ( tm->getTopFromReIntegrateList ( oldTransform, newTransform, frameIdx ) )
        {
            auto& f = g_imageManager->getIntegrateFrame ( frameIdx );
            DepthCameraData depthCameraData ( f.getDepthFrameGPU(), f.getColorFrameGPU() );
            MLIB_ASSERT ( !isnan ( oldTransform[0] ) && !isnan ( newTransform[0] ) && oldTransform[0] != -std::numeric_limits<float>::infinity() && newTransform[0] != -std::numeric_limits<float>::infinity() );
            deIntegrate ( depthCameraData, oldTransform );
            integrate ( depthCameraData, newTransform );
            tm->confirmIntegration ( frameIdx );
            continue;
        }
        else
        {
            break; //no more work to do
        }
    }
    g_sceneRep->garbageCollect();
}

void StopScanningAndExtractIsoSurfaceMC ( const std::string& filename, bool overwriteExistingFile /*= false*/ )
{
    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
    if ( GlobalAppState::get().s_sensorIdx == 7 ) //! hack for structure sensor
    {
        std::cout << "[marching cubes] stopped receiving frames from structure sensor" << std::endl;
        g_RGBDSensor->stopReceivingFrames();
    }
    std::cout << "running marching cubes...1" << std::endl;

    Timer t;


    g_marchingCubesHashSDF->clearMeshBuffer();
    if ( !GlobalAppState::get().s_streamingEnabled )
    {
        //g_chunkGrid->stopMultiThreading();
        //g_chunkGrid->streamInToGPUAll();
        g_marchingCubesHashSDF->extractIsoSurface ( g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData() );
        //g_chunkGrid->startMultiThreading();
    }
    else
    {
        vec4f posWorld = vec4f ( g_lastRigidTransform*GlobalAppState::get().s_streamingPos, 1.0f ); // trans lags one frame
        vec3f p ( posWorld.x, posWorld.y, posWorld.z );
        g_marchingCubesHashSDF->extractIsoSurface ( *g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius );
    }

    const mat4f& rigidTransform = mat4f::identity();//g_lastRigidTransform
    g_marchingCubesHashSDF->saveMesh ( filename, &rigidTransform, overwriteExistingFile );

    std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

    //g_sceneRep->debugHash();
    //g_chunkGrid->debugCheckForDuplicates();
}

void StopScanningAndExit ( bool aborted = false )
{
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
    g_depthSensingBundler->finishCorrespondenceEvaluatorLogging();
    std::vector<mat4f> trajectory;
    g_depthSensingBundler->getTrajectoryManager()->getOptimizedTransforms ( trajectory );
    if ( GlobalAppState::get().s_sensorIdx == 8 ) ( ( SensorDataReader* ) g_depthSensingRGBDSensor )->evaluateTrajectory ( trajectory );
#endif
#ifdef PRINT_MEM_STATS
    unsigned int heapOccCount = g_sceneRep->getHashParams().m_numSDFBlocks - g_sceneRep->getHeapFreeCount();
    std::cout << "=============== RECONSTRUCTION ===============" << std::endl;
    std::cout << "#hash buckets = " << g_sceneRep->getHashParams().m_hashNumBuckets << std::endl;
    std::cout << "#voxel blocks = " << heapOccCount << std::endl;
    std::cout << "=============== OPTIMIZATION ===============" << std::endl;
    g_depthSensingBundler->printMemStats();
#endif
    std::cout << "[ stop scanning and exit ]" << std::endl;
    if ( !aborted )
    {
        //estimate validity of reconstruction
        bool valid = true;
        unsigned int heapFreeCount = g_sceneRep->getHeapFreeCount();
        if ( heapFreeCount < 800 ) valid = false; // probably a messed up reconstruction (used up all the heap...)
        unsigned int numValidTransforms = 0, numTransforms = 0;
        //write trajectory
        const std::string saveFile = GlobalAppState::get().s_binaryDumpSensorFile;
        std::vector<mat4f> trajectory;
        g_bundler->getTrajectoryManager()->getOptimizedTransforms ( trajectory );
        numValidTransforms = PoseHelper::countNumValidTransforms ( trajectory );
        numTransforms = ( unsigned int ) trajectory.size();
        if ( numValidTransforms < ( unsigned int ) std::round ( 0.5f * numTransforms ) ) valid = false; // not enough valid transforms
        std::cout << "#VALID TRANSFORMS = " << numValidTransforms << std::endl;
        ( ( SensorDataReader* ) g_RGBDSensor )->saveToFile ( GlobalAppState::get().s_binaryDumpSensorFile + "sequence.sens", trajectory ); //overwrite the original file

        //if (GlobalAppState::get().s_sensorIdx == 8) ((SensorDataReader*)g_depthSensingRGBDSensor)->evaluateTrajectory(trajectory);

        //save ply
        std::cout << "[marching cubes] ";
        StopScanningAndExtractIsoSurfaceMC ( GlobalAppState::get().s_binaryDumpSensorFile + "scan_last.ply", true ); //force overwrite and existing plys
        //StopScanningAndExtractIsoSurfaceMC("debug/" + util::removeExtensions(util::fileNameFromPath(GlobalAppState::get().s_binaryDumpSensorFile)) + ".ply", true);
        std::cout << "done!" << std::endl;
        //write out confirmation file
        std::ofstream s ( GlobalAppState::get().s_binaryDumpSensorFile + "processed.txt" );
        if ( valid )  s << "valid = true" << std::endl;
        else		s << "valid = false" << std::endl;
        s << "heapFreeCount = " << heapFreeCount << std::endl;
        s << "numValidOptTransforms = " << numValidTransforms << std::endl;
        s << "numTransforms = " << numTransforms << std::endl;
        s.close();
    }
    else
    {
        std::ofstream s ( GlobalAppState::get().s_generateMeshDir + "processed.txt" );
        s << "valid = false" << std::endl;
        s << "ABORTED" << std::endl; // can only be due to invalid first chunk (i think)
        s.close();
    }
    fflush ( stdout );
}

void ResetDepthSensing()
{
    g_sceneRep->reset();
    //g_Camera.Reset();
    if ( g_chunkGrid )
    {
        g_chunkGrid->reset();
    }
}
