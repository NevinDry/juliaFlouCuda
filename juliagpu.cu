#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "juliagpu.h"
#include "cpu_bitmap.h"


#define DIM 300

struct cuComplexgpu {
	float r;
	float i;
	__device__ cuComplexgpu (float a, float b) : r(a), i(b){}
	__device__ float magnitude2(void) { return r * r + i * i; }
	__device__ cuComplexgpu operator*(const cuComplexgpu& a) {
		return cuComplexgpu(r * a.r - i * a.i, i * a.r + r*a.i);
	}
	__device__ cuComplexgpu operator+(const cuComplexgpu& a) {
		return cuComplexgpu(r + a.r, i + a.i);
	}
};

__device__ int juliagpu (int x, int y ) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM / 2);
	float jy = scale * (float)(DIM/2 - y)/(DIM / 2);

	cuComplexgpu c(-0.8, 0.156);
	cuComplexgpu a(jx, jy);

	for (int i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;		
	}

	return 1;
}

__global__ void kernelgpu (unsigned char *ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	int juliaValue = juliagpu(x, y);
	ptr[offset * 4 + 0] = 0;
	ptr[offset * 4 + 1] = 255 * juliaValue;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
	
}

__device__ int getIndex(int x,int y){
	return (y*DIM*4) + (x*4);
}


__global__ void blur(unsigned char *ptr){

	int x = blockIdx.x;
	int y = blockIdx.y;
	int i = 0;
	int moyenneCouleur[3] = {0,0,0};
	int cpt = 0;

	
	//on r�cup�re l'index et la couleur pour chaque block de pixel
	for (int i = -4; i<5;i++){
		for (int j = -4; j<5; j++){
			if(i+x >= 0 && j+y >= 0 && x+i < DIM && y+j < DIM){
				int index = getIndex(i+x,j+y);
				moyenneCouleur[0] += ptr[index];
				moyenneCouleur[1] += ptr[index + 1];
				moyenneCouleur[2] += ptr[index + 2];
				cpt++;
			}
		}
	}

	//synchro des threads 
	 __syncthreads();

    //on r��crit les couleurs
	int index = getIndex(x,y);
	ptr[index] = (moyenneCouleur[0] / cpt);
	ptr[index+1] = (moyenneCouleur[1] / cpt);
	ptr[index+2] = (moyenneCouleur[2] / cpt);
	ptr[index+3] = 255;
}


void main () {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());

	dim3 grid(DIM, DIM);
	kernelgpu<<<grid, 1>>>(dev_bitmap);
	
	blur<<<grid, 1>>>(dev_bitmap);
	
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);	

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
}
