#pragma once

#include "GlobalAppState.h"
#include "VoxelUtilHashSDF.h"
#include "MarchingCubesSDFUtil.h"
#include "CUDASceneRepChunkGrid.h"

class CUDAMarchingCubesHashSDF
{
public:
	CUDAMarchingCubesHashSDF(const MarchingCubesParams& params) {
		create(params);
	}

	~CUDAMarchingCubesHashSDF(void) {
		destroy();
	}

	static MarchingCubesParams parametersFromGlobalAppState(const GlobalAppState& gas) {
		MarchingCubesParams params;
		params.m_maxNumTriangles = gas.s_marchingCubesMaxNumTriangles;
		params.m_threshMarchingCubes = gas.s_SDFMarchingCubeThreshFactor*gas.s_SDFVoxelSize;
		params.m_threshMarchingCubes2 = gas.s_SDFMarchingCubeThreshFactor*gas.s_SDFVoxelSize;
		params.m_sdfBlockSize = SDF_BLOCK_SIZE;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashNumBuckets = gas.s_hashNumBuckets;
		
		std::cout<<"============================================="<<std::endl;
		std::cout<<"==" <<params.m_maxNumTriangles << std::endl;
		std::cout<<"==" <<params.m_threshMarchingCubes << std::endl;
		std::cout<<"==" <<params.m_threshMarchingCubes2 << std::endl;
		std::cout<<"==" <<params.m_sdfBlockSize << std::endl;
		std::cout<<"==" <<params.m_hashBucketSize << std::endl;
		std::cout<<"==" <<params.m_hashNumBuckets << std::endl;
		
		std::cout<<"============================================="<<std::endl;
		return params;
	}
	
	
	void clearMeshBuffer(void) {
		m_meshData.clear();
	}

	//! copies the intermediate result of extract isoSurfaceCUDA to the CPU and merges it with meshData
	void copyTrianglesToCPU();
	
	MarchingCubesData* copyMarchingCubeToGPU();
	
	void saveMesh(const std::string& filename, const mat4f *transform = NULL, bool overwriteExistingFile = false);

	void extractIsoSurface(const HashDataStruct& hashData, const HashParams& hashParams, const RayCastData& rayCastData, const vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), const vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);
	
	MarchingCubesData* extractIsoSurfaceGPUNoChrunk(const HashDataStruct& hashData, const HashParams& hashParams, const RayCastData& rayCastData, const vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), const vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);

	//void extractIsoSurfaceCPU(const HashData& hashData, const HashParams& hashParams, const RayCastData& rayCastData);

	void extractIsoSurface(CUDASceneRepChunkGrid& chunkGrid, const RayCastData& rayCastData, const vec3f& camPos, float radius);
	
	void extractISOSurfaceGPU (const HashDataStruct& hashData, const HashParams& hashParams, const RayCastData& rayCastData, const vec3f& minCorner = vec3f(0.0f, 0.0f, 0.0f), const vec3f& maxCorner = vec3f(0.0f, 0.0f, 0.0f), bool boxEnabled = false);
	
	MarchingCubesData* extractISOSurfaceGPU(CUDASceneRepChunkGrid& chunkGrid, const RayCastData& rayCastData, const vec3f& camPos, float radius);

	MeshDataf* getMeshData(){return &m_meshData;}

private:
	
	void create(const MarchingCubesParams& params);
	void destroy(void);

	MarchingCubesParams m_params;
	MarchingCubesData	m_data;

	MeshDataf m_meshData;

	Timer m_timer;
};

