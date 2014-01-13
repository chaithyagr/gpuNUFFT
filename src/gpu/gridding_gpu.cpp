#include "gridding_gpu.hpp"
#include "gridding_kernels.hpp"
#include "cufft_config.hpp"

#include "cuda_utils.hpp"

// ----------------------------------------------------------------------------
// gridding3D_gpu: NUFFT
//
// Inverse gridding implementation - interpolation from uniform grid data onto 
//                                   nonuniform k-space data based on optimized 
//                                   gridding kernel with minimal oversampling
//                                   ratio (see Beatty et al.)
//
// Basic steps: - apodization correction
//              - zero padding with osf
//              - FFT
//							- convolution and resampling
//
// parameters:
//	* data		     : output kspace data 
//  * data_count   : number of samples on trajectory
//  * n_coils      : number of channels or coils
//  * crds         : coordinates on trajectory, passed as SoA
//  * imdata       : input image data
//  * imdata_count : number of image data points
//  * grid_width   : size of grid 
//  * kernel       : precomputed convolution kernel as lookup table
//  * kernel_count : number of kernel lookup table entries
//  * sectors      : mapping of data indices according to each sector
//  * sector_count : number of sectors
//  * sector_centers: coordinates (x,y,z) of sector centers
//  * sector_width : width of sector 
//  * im_width     : dimension of image
//  * osr          : oversampling ratio
//  * gridding_out : enum indicating how far gridding has to be processed
//  
void gridding3D_gpu(CufftType**	data,			
                    int			data_count,
                    int			n_coils,
                    DType*		crds,	
                    DType2*		imdata,
                    int			imdata_count,
                    int			grid_width,
                    DType*		kernel,
                    int			kernel_count,
                    int			kernel_width,
                    IndType*	sectors,
                    int			sector_count,
                    IndType*	sector_centers,
                    int			sector_width,
                    int			im_width,
                    DType		osr,
                    const GriddingOutput gridding_out)
{
	showMemoryInfo();
	GriddingInfo* gi_host = initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,grid_width,im_width,osr,data_count);

	//cuda mem allocation
	DType2 *imdata_d;
	DType*crds_d, *deapo_d;
	CufftType *gdata_d, *data_d;
	IndType* sector_centers_d, *sectors_d;
	
	if (DEBUG)
		printf("allocate and copy imdata of size %d...\n",imdata_count);
	allocateDeviceMem<DType2>(&imdata_d,imdata_count);

	if (DEBUG)
		printf("allocate and copy gdata of size %d...\n",gi_host->grid_width_dim );
	allocateAndSetMem<CufftType>(&gdata_d, gi_host->grid_width_dim,0);

	if (DEBUG)
		printf("allocate and copy data of size %d...\n",data_count);
	allocateDeviceMem<CufftType>(&data_d,data_count);

	if (DEBUG)
		printf("allocate and copy coords of size %d...\n",3*data_count);
	allocateAndCopyToDeviceMem<DType>(&crds_d,crds,3*data_count);
	
	if (DEBUG)
		printf("allocate and copy kernel in const memory of size %d...\n",kernel_count);
	//HANDLE_ERROR(cudaMemcpyToSymbol(KERNEL,(void*)kernel,kernel_count*sizeof(DType)));
	initConstSymbol("KERNEL",(void*)kernel,kernel_count*sizeof(DType));

	if (DEBUG)
		printf("allocate and copy sectors of size %d...\n",sector_count+1);
	allocateAndCopyToDeviceMem<IndType>(&sectors_d,sectors,sector_count+1);
	if (DEBUG)
		printf("allocate and copy sector_centers of size %d...\n",3*sector_count);
	allocateAndCopyToDeviceMem<IndType>(&sector_centers_d,sector_centers,3*sector_count);
	if (n_coils > 1)
	{
		if (DEBUG)
			printf("allocate and precompute deapofunction of size %d...\n",imdata_count);
		allocateDeviceMem<DType>(&deapo_d,imdata_count);
		precomputeDeapodization(deapo_d,gi_host);
	}
	
	if (DEBUG)
		printf("sector pad width: %d\n",gi_host->sector_pad_width);
	
	//Inverse fft plan and execution
	cufftHandle fft_plan;
	if (DEBUG)
		printf("creating cufft plan with %d,%d,%d dimensions\n",gi_host->grid_width,gi_host->grid_width,gi_host->grid_width);
	cufftResult res = cufftPlan3d(&fft_plan, gi_host->grid_width,gi_host->grid_width,gi_host->grid_width, CufftTransformType) ;
	if (res != CUFFT_SUCCESS) 
		printf("error on CUFFT Plan creation!!! %d\n",res);
	int err;

	//iterate over coils and compute result
	for (int coil_it = 0; coil_it < n_coils; coil_it++)
	{
		int data_coil_offset = coil_it * data_count;
		int im_coil_offset = coil_it * imdata_count;//gi_host->width_dim;
		//reset temp array
		copyToDevice(imdata + im_coil_offset,imdata_d,imdata_count);	
		cudaMemset(data_d,0, sizeof(CufftType)*data_count);
		cudaMemset(gdata_d,0, sizeof(CufftType)*gi_host->grid_width_dim);

		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));
		// apodization Correction
		if (n_coils > 1 && deapo_d != NULL)
			performForwardDeapodization(imdata_d,deapo_d,gi_host);
		else
		  performForwardDeapodization(imdata_d,gi_host);
		
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
	  // resize by oversampling factor and zero pad
		performPadding(imdata_d,gdata_d,gi_host);
	
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 3: %s\n",cudaGetErrorString(cudaGetLastError()));
		// shift image to get correct zero frequency position
		performFFTShift(gdata_d,INVERSE,gi_host->grid_width);
	
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 4: %s\n",cudaGetErrorString(cudaGetLastError()));
		// eventually free imdata_d
		// Forward FFT to kspace domain
		if (err=pt2CufftExec(fft_plan, gdata_d, gdata_d, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			fprintf(stderr,"cufft has failed with err %i \n",err);
			showMemoryInfo(true,stderr);
		}
		
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 5: %s\n",cudaGetErrorString(cudaGetLastError()));
		performFFTShift(gdata_d,FORWARD,gi_host->grid_width);
		
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 6: %s\n",cudaGetErrorString(cudaGetLastError()));
		// convolution and resampling to non-standard trajectory
		performForwardConvolution(data_d,crds_d,gdata_d,NULL,sectors_d,sector_centers_d,gi_host);
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 7: %s\n",cudaGetErrorString(cudaGetLastError()));

		performFFTScaling(data_d,gi_host->data_count,gi_host);
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error: at adj  thread synchronization 8: %s\n",cudaGetErrorString(cudaGetLastError()));

		//get result
		copyFromDevice<CufftType>(data_d, *data + data_coil_offset,data_count);
	}//iterate over coils
	cufftDestroy(fft_plan);
	// Destroy the cuFFT plan.
	if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at thread synchronization 9: %s\n",cudaGetErrorString(cudaGetLastError()));
	freeTotalDeviceMemory(data_d,crds_d,gdata_d,imdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop
	if (n_coils > 1 && deapo_d != NULL)
		cudaFree(deapo_d);

	if ((cudaThreadSynchronize() != cudaSuccess))
		fprintf(stderr,"error in gridding3D_gpu function: %s\n",cudaGetErrorString(cudaGetLastError()));
	free(gi_host);
}

// ----------------------------------------------------------------------------
// gridding3D_gpu_adj: NUFFT^H
//
// Gridding implementation - interpolation from nonuniform k-space data onto 
//                           oversampled grid based on optimized gridding kernel
//                           with minimal oversampling ratio (see Beatty et al.)
//
// Basic steps: - density compensation
//              - convolution with interpolation function
//              - iFFT
//              - cropping due to oversampling ratio
//              - apodization correction
//
// parameters:
//	* data		     : input kspace data 
//  * data_count   : number of samples on trajectory
//  * n_coils      : number of channels or coils
//  * crds         : coordinate array on trajectory
//  * imdata       : output image data
//  * imdata_count : number of image data points
//  * grid_width   : size of grid 
//  * kernel       : precomputed convolution kernel as lookup table
//  * kernel_count : number of kernel lookup table entries
//  * sectors      : mapping of start and end points of each sector
//  * sector_count : number of sectors
//  * sector_centers: coordinates (x,y,z) of sector centers
//  * sector_width : width of sector 
//  * im_width     : dimension of image
//  * osr          : oversampling ratio
//  * do_comp      : true, if density compensation has to be done
//  * density_comp : densiy compensation array
//  * gridding_out : enum indicating how far gridding has to be processed
//  
void gridding3D_gpu_adj(DType2*		data,			
                        int			data_count,
                        int			n_coils,
                        DType*		crds,
                        CufftType**	imdata,
                        IndType		imdata_count,
                        int			grid_width,
                        DType*		kernel,
                        int			kernel_count,
                        int			kernel_width,
                        IndType*	sectors,
                        int			sector_count,
                        IndType*	sector_centers,
                        int			sector_width,
                        int			im_width,
                        DType		osr,
                        bool		do_comp,
                        DType*  density_comp,
                        const GriddingOutput gridding_out)
{
	assert(sectors != NULL);
	
	showMemoryInfo();

	//split and run sectors into blocks
	//and each data point to one thread inside this block 
	GriddingInfo* gi_host = initAndCopyGriddingInfo(sector_count,sector_width,kernel_width,kernel_count,grid_width,im_width,osr,data_count);
	
	DType2* data_d;
	DType* crds_d, *density_comp_d, *deapo_d;
	CufftType *gdata_d, *imdata_d;
	IndType* sector_centers_d, *sectors_d;
	
	if (DEBUG)
		printf("allocate and copy imdata of size %d...\n",imdata_count);
	allocateAndCopyToDeviceMem<CufftType>(&imdata_d,*imdata,imdata_count);//Konvention!!!

	if (DEBUG)
		printf("allocate and copy gdata of size %d...\n",gi_host->grid_width_dim);
	allocateDeviceMem<CufftType>(&gdata_d,gi_host->grid_width_dim);

	if (DEBUG)
		printf("allocate and copy data of size %d...\n",data_count);
	allocateDeviceMem<DType2>(&data_d,data_count);

	if (DEBUG)
		printf("allocate and copy coords of size %d...\n",3*data_count);
	allocateAndCopyToDeviceMem<DType>(&crds_d,crds,3*data_count);
	
	if (DEBUG)
		printf("allocate and copy kernel in const memory of size %d...\n",kernel_count);
	
	initConstSymbol("KERNEL",(void*)kernel,kernel_count*sizeof(DType));

	//allocateAndCopyToDeviceMem<DType>(&kernel_d,kernel,kernel_count);
	if (DEBUG)
		printf("allocate and copy sectors of size %d...\n",sector_count+1);
	allocateAndCopyToDeviceMem<IndType>(&sectors_d,sectors,sector_count+1);
	if (DEBUG)
		printf("allocate and copy sector_centers of size %d...\n",3*sector_count);
	allocateAndCopyToDeviceMem<IndType>(&sector_centers_d,sector_centers,3*sector_count);
	
	if (do_comp == true)	
	{
		if (DEBUG)
			printf("allocate and copy density compensation of size %d...\n",data_count);
		allocateAndCopyToDeviceMem<DType>(&density_comp_d,density_comp,data_count);
	}
	
	if (n_coils > 1)
	{
		if (DEBUG)
			printf("allocate precompute deapofunction of size %d...\n",imdata_count);
		allocateDeviceMem<DType>(&deapo_d,imdata_count);
		precomputeDeapodization(deapo_d,gi_host);
	}
	if (DEBUG)
		printf("sector pad width: %d\n",gi_host->sector_pad_width);
	
	//Inverse fft plan and execution
	cufftHandle fft_plan;
	if (DEBUG)
		printf("creating cufft plan with %d,%d,%d dimensions\n",gi_host->grid_width,gi_host->grid_width,gi_host->grid_width);
	cufftResult res = cufftPlan3d(&fft_plan, gi_host->grid_width,gi_host->grid_width,gi_host->grid_width, CufftTransformType) ;
	if (res != CUFFT_SUCCESS) 
		fprintf(stderr,"error on CUFFT Plan creation!!! %d\n",res);
	int err;

	//iterate over coils and compute result
	for (int coil_it = 0; coil_it < n_coils; coil_it++)
	{
		int data_coil_offset = coil_it * data_count;
		int im_coil_offset = coil_it * imdata_count;//gi_host->width_dim;

		cudaMemset(gdata_d,0, sizeof(CufftType)*gi_host->grid_width_dim);
		copyToDevice(data + data_coil_offset, data_d,data_count);
	
		if (do_comp == true)
			performDensityCompensation(data_d,density_comp_d,gi_host);
		
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 1: %s\n",cudaGetErrorString(cudaGetLastError()));
		performConvolution(data_d,crds_d,gdata_d,NULL,sectors_d,sector_centers_d,gi_host);
	
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			fprintf(stderr,"error at adj  thread synchronization 2: %s\n",cudaGetErrorString(cudaGetLastError()));
		if (gridding_out == CONVOLUTION)
		{
			if (DEBUG)
				printf("stopping output after CONVOLUTION step\n");
			//get output
			copyFromDevice<CufftType>(gdata_d,*imdata,gi_host->grid_width_dim);
			if (DEBUG)
				printf("test value at point zero: %f\n",(*imdata)[0].x);
			freeTotalDeviceMemory(data_d,crds_d,imdata_d,gdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop token

			free(gi_host);
			// Destroy the cuFFT plan.
			cufftDestroy(fft_plan);
			return;
		}
		if ((cudaThreadSynchronize() != cudaSuccess))
			fprintf(stderr,"error at adj thread synchronization 3: %s\n",cudaGetErrorString(cudaGetLastError()));
		performFFTShift(gdata_d,INVERSE,gi_host->grid_width);
		
		//Inverse FFT
		if (err=pt2CufftExec(fft_plan, gdata_d, gdata_d, CUFFT_INVERSE) != CUFFT_SUCCESS)
		{
			fprintf(stderr,"cufft has failed at adj with err %i \n",err);
			showMemoryInfo(true,stderr);
		}
	  	if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			 fprintf(stderr,"error at adj thread synchronization 4: %s\n",cudaGetErrorString(cudaGetLastError()));
	
		if (gridding_out == FFT)
		{
			if (DEBUG)
				printf("stopping output after FFT step\n");
			//get output
			copyFromDevice<CufftType>(gdata_d,*imdata,gi_host->grid_width_dim);
			
			//free memory
			if (cufftDestroy(fft_plan) != CUFFT_SUCCESS)
				printf("error on destroying cufft plan\n");
			freeTotalDeviceMemory(data_d,crds_d,imdata_d,gdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop token
			free(gi_host);
			// Destroy the cuFFT plan.
			printf("last cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
			return;
		}
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 5: %s\n",cudaGetErrorString(cudaGetLastError()));
		performFFTShift(gdata_d,INVERSE,gi_host->grid_width);
			
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 6: %s\n",cudaGetErrorString(cudaGetLastError()));
		performCrop(gdata_d,imdata_d,gi_host);
		
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 7: %s\n",cudaGetErrorString(cudaGetLastError()));
		//check if precomputed deapo function can be used
		if (n_coils > 1 && deapo_d != NULL)
			performDeapodization(imdata_d,deapo_d,gi_host);
		else
		  performDeapodization(imdata_d,gi_host);
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error at adj thread synchronization 8: %s\n",cudaGetErrorString(cudaGetLastError()));
	
		performFFTScaling(imdata_d,gi_host->im_width_dim,gi_host);
		if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
			printf("error: at adj  thread synchronization 9: %s\n",cudaGetErrorString(cudaGetLastError()));
	
		//get result
		copyFromDevice<CufftType>(imdata_d,*imdata+im_coil_offset,imdata_count);
	}//iterate over coils
	if (DEBUG && (cudaThreadSynchronize() != cudaSuccess))
      printf("error: at adj  thread synchronization 10: %s\n",cudaGetErrorString(cudaGetLastError()));
	// Destroy the cuFFT plan.
	cufftDestroy(fft_plan);
	freeTotalDeviceMemory(data_d,crds_d,gdata_d,imdata_d,sectors_d,sector_centers_d,NULL);//NULL as stop
	if (do_comp == true)
		cudaFree(density_comp_d);
	if (n_coils > 1)
		cudaFree(deapo_d);
	
	if ((cudaThreadSynchronize() != cudaSuccess))
		fprintf(stderr,"error in gridding3D_gpu_adj function: %s\n",cudaGetErrorString(cudaGetLastError()));
	free(gi_host);
}