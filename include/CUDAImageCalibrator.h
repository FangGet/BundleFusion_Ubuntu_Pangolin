#pragma once

#include "RGBDSensor.h"
#include "GlobalAppState.h"

#include <cuda_runtime.h> 


class CUDAImageCalibrator
{
	public:

		CUDAImageCalibrator() { d_dummyColor = NULL; }
		~CUDAImageCalibrator() {}

		void OnD3D11DestroyDevice();

	private:

		unsigned int m_width, m_height;

		float4*						d_dummyColor; // don't need to render color but the rgbdrenderer expects a float4 color array...

		Timer m_timer;
};
