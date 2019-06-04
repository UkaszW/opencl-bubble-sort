//
//  main.cpp
//  opencl-bubble-sort
//
//  Created by Łukasz Wójcik on 02/06/2019.
//  Copyright © 2019 Łukasz Wójcik. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_STATUS( status, message )   \
if(status != CL_SUCCESS) \
{ \
printf( message); \
printf( "\n ERR %d\n", status  ); \
fflush(NULL);\
return 1; \
}

#define MATRIX_SIZE 16384 // matrix square dimension
#define LOCAL_SIZE 8 // for GPU best: 8 CPU: 1

#define MAX_SOURCE_SIZE (1000000)

#define CL_SILENCE_DEPRECATION

// Allocates a matrix with random float entries.
void randomInit(cl_float* data, int size)
{
    srand(0);
    for (int i = 0; i < size; i++)
    {
        data[i] = rand() / (cl_float) RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    //------------------------------------------------------------
    //defining the input argument
    //------------------------------------------------------------
    cl_int numData = MATRIX_SIZE;
    
    printf("Tab size: %d \n", numData);
    
    //-------------------------------------------------------------
    //first we have to get the platform we have in hand
    //-------------------------------------------------------------
    cl_int ret;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_uint num_platforms;
    cl_uint num_entries = 1;
    char str[1024];
    size_t size;
    cl_uint tmp;
    cl_ulong utmp;
    cl_uint num_devices;
    
    // Device info.
    ret = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 1024, str, &size);
    if (ret == CL_SUCCESS) {
        std::cout << "Platform name      : " << str << std::endl;
    } else {
        printf("Error: getting platfotm info \n");
    }
    ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_entries, &device, &num_devices);
    CHECK_STATUS(ret, "Error: getting device id");
    cl_device_type t;
    ret = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(t), &t, &size);
    if (ret == CL_SUCCESS) {
        if (t == CL_DEVICE_TYPE_CPU) {
            std::cout << "Device type        : CPU" << std::endl;
        }
        if (t == CL_DEVICE_TYPE_GPU) {
            std::cout << "Device type        : GPU" << std::endl;
        }
    } else {
        printf("Error: getting platfotm info \n");
    }
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(tmp), &tmp, &size);
    if (ret == CL_SUCCESS) {
        std::cout << "Number of units    : " << tmp << std::endl;
    } else {
        printf("Error: getting platfotm info \n");
    }
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(utmp), &utmp, &size);
    if (ret == CL_SUCCESS) {
        std::cout << "Max. memory alloc. : " << utmp / (1024 * 1024) << " MB" << std::endl;
    } else {
        printf("Error: getting platfotm info \n");
    }
    ret = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(utmp), &utmp, &size);
    if (ret == CL_SUCCESS) {
        std::cout << "Global mem. size   : " << utmp / (1024 * 1024) << " MB" << std::endl;
    } else {
        printf("Error: getting platfotm info \n");
    }
    ret = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(tmp), &tmp, &size);
    if (ret == CL_SUCCESS) {
        std::cout << "Clock frequency    : " << tmp << " MHz " << std::endl;
    } else {
        printf("Error: getting platfotm info \n");
    }
    
    //----------------------------------------------------------------
    // creating context and Command Queue
    //----------------------------------------------------------------
    cl_context context;
    context = clCreateContext(NULL, num_devices, &device, NULL, NULL, &ret);
    
    CHECK_STATUS( ret, "Error: in Creating Context \n");
    
    // creating command queue
    cl_command_queue cq;
    
    cq = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
    
    CHECK_STATUS (ret , "Error: in Creating command Queue \n");
    
    cl_event event;
    
    //------------------------------------------------------------------------------
    // Load the kernel, creating the program, Build the program and create
    //-------------------------------------------------------------------------------
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen("/Users/lukaszwojcik/Development/opencl-bubble-sort/opencl-bubble-sort/bs_kernel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load the kernel \n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose (fp);
    
    
    // creating a program with source
    cl_program program;
    //fprintf (stderr, "%s",source_str);
    program = clCreateProgramWithSource(context, 1, (const char **) &source_str,
                                        (const size_t *) &source_size, &ret);
    
    CHECK_STATUS(ret, "Error: in Creating The program \n");
    
    //Building the OpenCL program
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    CHECK_STATUS(ret,"Error: in Building The program \n");
    
    //creating the Kernel
    cl_kernel kernel;
    kernel = clCreateKernel(program, "bubble_sort", &ret);
    
    CHECK_STATUS(ret, "Error: in Creating The Kernel \n");
    
    //----------------------------------------------------------------------
    /* OpenCL buffers */
    //---------------------------------------------------------------------
    
    // creating the buffer in the HOST,
    size_t dim =  numData;
    size_t ldim = LOCAL_SIZE;
    cl_float *host_tab = (cl_float*)malloc(sizeof(cl_float) * numData);
    
    // initiating source buffer in host
    randomInit(host_tab, numData);
//    std::cout << "Input: ";
//    for (int x = 0; x < numData; x++) {
//        std::cout << host_tab[x] << " ";
//    }
//    std::cout << std::endl;
    
    // allocating source buffer in GPU
    cl_mem device_tab;
    device_tab = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              numData * sizeof(cl_float), NULL, &ret);
    if (ret != CL_SUCCESS)    printf("Error: in allocating buffer A in GPU \n");
    
    // copy source buffer into GPU
    ret = clEnqueueWriteBuffer (cq, device_tab, CL_TRUE, 0, numData * sizeof(cl_float),
                                host_tab, 0, 0, &event );
    
    CHECK_STATUS(ret ,"Error: in copying source buffer into GPU \n");
    
    // setting the arguments
    ret = clSetKernelArg( kernel, 0, sizeof (cl_int), &numData); // 0 indicates the first argument
    if (ret != CL_SUCCESS)    printf("Error: setting the first argument \n");
    
    ret = clSetKernelArg( kernel, 1, sizeof (cl_mem), &device_tab); // 1 indicates the second argument
    if (ret != CL_SUCCESS)    printf("Error: setting the second argument \n");
    
    // main function for launching the kernel
    cl_uint dimension = 1;
    size_t global_work_size[1] = {dim};
    size_t local_work_size[1] = {ldim};
    ret = clEnqueueNDRangeKernel(cq, kernel, dimension , NULL, global_work_size, local_work_size,
                                 0, NULL, &event);
    if (ret != CL_SUCCESS) {
        printf("Error: Launching Kernel \n");
    } else {
        // measure time
        cl_ulong            time_end;
        cl_ulong            time_start;
        ret = clWaitForEvents(1, &event);
        if (ret != CL_SUCCESS)    printf("Error: Waiting events \n");
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        std::cout << "\nComputation time   : " << (time_end - time_start) / 1000 << " us" << std::endl;
    }
    
    // finish the execution
    ret = clFlush(cq);
    ret = clFinish(cq);
    
    if (ret != CL_SUCCESS)    printf("Error: Finishing the execution \n");
    
    //--------------------------------------------------------------------------
    /* Obtain the result from GPU to the CPU */
    //--------------------------------------------------------------------------
    
    // retrieving the buffer
    ret = clEnqueueReadBuffer (cq, device_tab, CL_TRUE, 0, numData * sizeof(cl_float),
                               host_tab, 0, NULL, &event);
    if (ret != CL_SUCCESS)    printf("Error: retrieving DST buffer into CPU \n");
    
    // Display the result to the screen
    
//    std::cout << "Result: ";
//    for (int x = 0; x < numData; x++) {
//        std::cout << host_tab[x] << " ";
//    }
//    std::cout << std::endl;
    
    fflush(NULL);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(device_tab);
    ret = clReleaseCommandQueue(cq);
    ret = clReleaseContext(context);
    
    fflush(NULL);
    
    return(0);
}
