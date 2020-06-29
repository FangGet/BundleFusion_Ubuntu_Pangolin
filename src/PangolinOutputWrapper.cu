#include <VoxelUtilHashSDF.h>
#include <RayCastSDFUtil.h>
#include <MarchingCubesSDFUtil.h>

__global__ void convertMarchCubeToGLArray ( float3* march_cube, float3* dVertexArray, uchar3* dColourArray, uint3* dIndicesArray )
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    dVertexArray[3*x + 0] = make_float3 ( march_cube[6 * x + 0].x,march_cube[6 * x + 0].y,march_cube[6 * x + 0].z );
    dVertexArray[3*x + 1] = make_float3 ( march_cube[6 * x + 2].x,march_cube[6 * x + 2].y,march_cube[6 * x + 2].z );
    dVertexArray[3*x + 2] = make_float3 ( march_cube[6 * x + 4].x,march_cube[6 * x + 4].y,march_cube[6 * x + 4].z );

    if ( dColourArray != nullptr ) {
        dColourArray[3*x + 0].x = 255.f * march_cube[6 * x + 1].x;
        dColourArray[3*x + 0].y = 255.f * march_cube[6 * x + 1].y;
        dColourArray[3*x + 0].z = 255.f * march_cube[6 * x + 1].z;

        dColourArray[3*x + 1].x = 255.f * march_cube[6 * x + 3].x;
        dColourArray[3*x + 1].y = 255.f * march_cube[6 * x + 3].y;
        dColourArray[3*x + 1].z = 255.f * march_cube[6 * x + 3].z;

        dColourArray[3*x + 2].x = 255.f * march_cube[6 * x + 5].x;
        dColourArray[3*x + 2].y = 255.f * march_cube[6 * x + 5].y;
        dColourArray[3*x + 2].z = 255.f * march_cube[6 * x + 5].z;
    }

    if ( dIndicesArray != nullptr ) {
        dIndicesArray[x] = make_uint3 ( 3*x +0, 3*x + 1, 3*x + 2 );
    }
}

extern "C" void launch_convert_kernel ( float3* march_cube, float3* dVertexArray, uchar3* dColourArray, uint3* dIndicesArray,
                                        unsigned int size )
{
    dim3 block ( 64 );
    dim3 grid ( size / 64 );
    convertMarchCubeToGLArray<<< grid, block>>> ( march_cube,dVertexArray,dColourArray,dIndicesArray );
}
