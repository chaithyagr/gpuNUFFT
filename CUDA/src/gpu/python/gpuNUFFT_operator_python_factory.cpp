/*
This file holds the python bindings for gpuNUFFT library.
Authors:
Chaithya G R <chaithyagr@gmail.com>
Carole Lazarus <carole.m.lazarus@gmail.com>
*/

#ifndef GPUNUFFT_OPERATOR_PYTHON_FACTORY_H_INCLUDED
#define GPUNUFFT_OPERATOR_PYTHON_FACTORY_H_INCLUDED
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h>
#include <cublas.h>
#include "config.hpp"
#include "gpuNUFFT_operator_factory.hpp"
#include <algorithm>  // std::sort
#include <vector>     // std::vector
#include <string>

namespace py = pybind11;

template <typename TType>
gpuNUFFT::Array<TType>
readNumpyArray(py::array_t<TType> data)
{
    py::buffer_info myData = data.request();
    TType *t_data = (TType *) myData.ptr;
    gpuNUFFT::Array<TType> dataArray;
    dataArray.data = t_data;
    return dataArray;
}

gpuNUFFT::Array<DType2>
readNumpyArray(py::array_t<std::complex<DType>> data)
{
    gpuNUFFT::Array<DType2> dataArray;
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    DType2 *new_data = reinterpret_cast<DType2(&)[0]>(*t_data);
    dataArray.data = new_data;
    return dataArray;
}

gpuNUFFT::Array<DType2>
copyNumpyArray(py::array_t<std::complex<DType>> data, unsigned long alloc_size)
{
    gpuNUFFT::Array<DType2> dataArray;
    py::buffer_info myData = data.request();
    std::complex<DType> *t_data = (std::complex<DType> *) myData.ptr;
    DType2 *my_data = reinterpret_cast<DType2(&)[0]>(*t_data);
    DType2 *copy_data = (DType2 *) malloc(alloc_size*sizeof(DType2));
    memcpy(copy_data, my_data, alloc_size*sizeof(DType2));
    dataArray.data = copy_data;
    return dataArray;
}

class GpuNUFFTPythonOperator
{
    gpuNUFFT::GpuNUFFTOperatorFactory factory;
    gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp;
    int trajectory_length, n_coils, dimension, n_interpolators;
    bool has_sense_data;
    gpuNUFFT::Dimensions imgDims;
    // sensitivity maps
    gpuNUFFT::Array<DType2> sensArray;
    public:
    GpuNUFFTPythonOperator(py::array_t<DType> kspace_loc, py::array_t<int> image_size, int num_coils,
    py::array_t<std::complex<DType>> sense_maps,  py::array_t<float> density_comp, int kernel_width=3,
    int sector_width=8, int osr=2, bool balance_workload=1)
    {
        // k-space coordinates
        py::buffer_info sample_loc = kspace_loc.request();
        trajectory_length = sample_loc.shape[1];
        dimension = sample_loc.shape[0];
        gpuNUFFT::Array<DType> kSpaceTraj = readNumpyArray(kspace_loc);
        kSpaceTraj.dim.length = trajectory_length;

        // density compensation weights
        gpuNUFFT::Array<DType> density_compArray = readNumpyArray(density_comp);
        py::buffer_info interpolator_info = density_comp.request();
        n_interpolators = interpolator_info.shape[0];
        density_compArray.dim.length = trajectory_length * interpolator_info.shape[0];

        // image size
        py::buffer_info img_dim = image_size.request();
        int *dims = (int *) img_dim.ptr;
        imgDims.width = dims[0];
        imgDims.height = dims[1];
        if(dimension==3)
            imgDims.depth = dims[2];
        else
            imgDims.depth = 0;

        n_coils = num_coils;

        // sensitivity maps
        py::buffer_info sense_maps_buffer = sense_maps.request();
        if (sense_maps_buffer.shape[0] != n_interpolators)
        {
            printf("ERROR: Bad W0 matrix\n");
        }
        else
        {
            sensArray = copyNumpyArray(sense_maps, imgDims.count() * n_interpolators);
            sensArray.dim = imgDims;
            sensArray.dim.channels = n_interpolators;
            has_sense_data = true;
        }
        factory.setBalanceWorkload(balance_workload);
        gpuNUFFTOp = factory.createGpuNUFFTOperator(
            kSpaceTraj, density_compArray, sensArray, kernel_width, sector_width,
            osr, imgDims);
        cudaThreadSynchronize();
    }

    py::array_t<std::complex<DType>> op(py::array_t<std::complex<DType>> image)
    {
        py::array_t<std::complex<DType>> out_result({n_coils, trajectory_length});
        py::buffer_info out = out_result.request();
        std::complex<DType> *t_data = (std::complex<DType> *) out.ptr;
        DType2 *new_data = reinterpret_cast<DType2(&)[0]>(*t_data);
        gpuNUFFT::Array<CufftType> dataArray;
        dataArray.data = new_data;
        dataArray.dim.length = trajectory_length;
        dataArray.dim.channels = n_coils;

        gpuNUFFT::Array<DType2> imdataArray = readNumpyArray(image);
        imdataArray.dim = imgDims;
        imdataArray.dim.channels = n_coils;
        gpuNUFFTOp->performForwardGpuNUFFT(imdataArray, dataArray);
        cudaThreadSynchronize();
        return out_result;
    }
    py::array_t<std::complex<DType>> adj_op(py::array_t<std::complex<DType>> kspace_data)
    {
        int depth = imgDims.depth;
        if(dimension==2)
            depth = 1;
        py::array_t<std::complex<DType>> out_result;
        if(has_sense_data == false)
            out_result.resize({n_coils, depth, (int)imgDims.height, (int)imgDims.width});
        else
            out_result.resize({depth, (int)imgDims.height, (int)imgDims.width});
        py::buffer_info out = out_result.request();
        std::complex<DType> *t_data = (std::complex<DType> *) out.ptr;
        DType2 *new_data = reinterpret_cast<DType2(&)[0]>(*t_data);
        gpuNUFFT::Array<DType2> imdataArray;
        imdataArray.data = new_data;
        imdataArray.dim = imgDims;
        if(has_sense_data == false)
            imdataArray.dim.channels = n_coils;
        gpuNUFFT::Array<CufftType> dataArray = readNumpyArray(kspace_data);
        dataArray.dim.length = trajectory_length;
        dataArray.dim.channels = n_coils;
        gpuNUFFTOp->performGpuNUFFTAdj(dataArray, imdataArray);
        cudaThreadSynchronize();
        return out_result;
    }
    ~GpuNUFFTPythonOperator()
    {
        delete gpuNUFFTOp;
        if(has_sense_data == true)
            free(sensArray.data);
    }
};
PYBIND11_MODULE(gpuNUFFT, m) {
    py::class_<GpuNUFFTPythonOperator>(m, "NUFFTOp")
        .def(py::init<py::array_t<DType>, py::array_t<int>, int, py::array_t<std::complex<DType>>, py::array_t<float>, int, int, int, bool>())
        .def("op", &GpuNUFFTPythonOperator::op)
        .def("adj_op",  &GpuNUFFTPythonOperator::adj_op);
}
#endif  // GPUNUFFT_OPERATOR_MATLABFACTORY_H_INCLUDED
