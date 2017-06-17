#include <iostream>
//#include <cstdlib>
#include <omp.h>
//#include <stdio.h>
//#include <stdlib.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

cl_program load_program(cl_context context, const char* filename) {
    std::ifstream in(filename, std::ios_base::binary);
    if(!in.good()) {
        printf("err0\n");
        return 0;
    }

    // get file length
    in.seekg(0, std::ios_base::end);
    size_t length = in.tellg();
    in.seekg(0, std::ios_base::beg);

    // read program source
    std::vector<char> data(length + 1);
    in.read(&data[0], length);
    data[length] = 0;

    // create and build program 
    const char* source = &data[0];
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    if(program == 0) {
        printf("err1\n");
        return 0;
    }

    if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
        printf("err2\n");
        return 0;
    }

    return program;
}

int main(int argc, const char * argv[]) {
	double omp_get_wtime(void);
	double startCK, finishCK;

	startCK = omp_get_wtime();
	//////////////////////////////////////////////////////////////////
    //int height, width;
	uchar *data1;
    uchar *data2;
    uchar *data3;

    IplImage *Image1=cvLoadImage("image/mid11.png",1);
    IplImage *Image2=cvLoadImage("image/mid21.png",1);
    
    //IplImage *RedImage,*GreenImage,*BlueImage;
    //IplImage *Channel1,*Channel2,*Channel3;
    
    CvSize Size1=cvGetSize(Image1);

    IplImage *OutImage=cvCreateImage(Size1,IPL_DEPTH_8U,3);
    
    int inSize1[2]={Image1->height,Image1->width};
    int inSize2[2]={Image2->height,Image2->width};
    int *ins1 = (int *)inSize1;
    int *ins2 = (int *)inSize2;

	data1 = (uchar *)Image1->imageData;
    data2 = (uchar *)Image2->imageData;
    data3 = (uchar *)OutImage->imageData;

    const int DATA_SIZE1 = inSize1[0]*inSize1[1]*3;
    const int DATA_SIZE2 = inSize2[0]*inSize2[1]*3;
    size_t work_size = inSize1[0]*inSize1[1];

    std::cout << "h1, w1: " << inSize1[0] << ", " << inSize1[1] << "\n";
    std::cout << "h2, w2: " << inSize2[0] << ", " << inSize2[1] << "\n";
    std::cout << "DataSize 1: " << DATA_SIZE1 << "\n";
    std::cout << "DataSize 2: " << DATA_SIZE2 << "\n";
    std::cout << "WorkSize 1: " << work_size << "\n";
    
    cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);
   
/////////////////////////////////////////////////////////
	cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platforms\n";
        return 0;
    }

    std::vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platform ID\n";
        return 0;
    }

    ////////Device////////
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
    if(context == 0) {
        std::cerr << "Can't create OpenCL context\n";
        return 0;
    }

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    std::string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    std::cout << "Device: " << devname.c_str() << "\n";

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return 0;
    }

    cl_mem cl_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * DATA_SIZE1, &data1[0], NULL);
    cl_mem cl_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * DATA_SIZE2, &data2[0], NULL);
    cl_mem cl_size1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*2, &ins1[0], NULL);
    cl_mem cl_size2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*2, &ins2[0], NULL);
    cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * DATA_SIZE1, NULL, NULL);

    if(cl_in1 == 0 || cl_in2 == 0 || cl_size1 == 0 || cl_size2 == 0 || cl_res == 0) {
        std::cerr << "Can't create OpenCL buffer\n";
        clReleaseMemObject(cl_in1);
        clReleaseMemObject(cl_in2);
        clReleaseMemObject(cl_size1);
        clReleaseMemObject(cl_size2);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    cl_program program = load_program(context, "shader3.cl");
    if(program == 0) {
        std::cerr << "Can't load or build program\n";
        clReleaseMemObject(cl_in1);
        clReleaseMemObject(cl_in2);
        clReleaseMemObject(cl_size1);
        clReleaseMemObject(cl_size2);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    cl_kernel adder = clCreateKernel(program, "adder", 0);
    if(adder == 0) {
        std::cerr << "Can't load kernel\n";
        clReleaseProgram(program);
        clReleaseMemObject(cl_in1);
        clReleaseMemObject(cl_in2);
        clReleaseMemObject(cl_size1);
        clReleaseMemObject(cl_size2);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    ////////Set Kernel Arg////////
    clSetKernelArg(adder, 0, sizeof(cl_in1), &cl_in1);
    clSetKernelArg(adder, 1, sizeof(cl_in2), &cl_in2);
    clSetKernelArg(adder, 2, sizeof(cl_size1), &cl_size1);
    clSetKernelArg(adder, 3, sizeof(cl_size2), &cl_size2);
    clSetKernelArg(adder, 4, sizeof(cl_res), &cl_res);

    err = clEnqueueNDRangeKernel(queue, adder, 1, 0, &work_size, 0, 0, 0, 0);

    if(err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(uchar) * DATA_SIZE1, &data3[0], 0, 0, 0);
    }

    clReleaseKernel(adder);
    clReleaseProgram(program);
    clReleaseMemObject(cl_in1);
    clReleaseMemObject(cl_in2);
    clReleaseMemObject(cl_size1);
    clReleaseMemObject(cl_size2);
    clReleaseMemObject(cl_res);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    ////////////////////////////////////////////////
    printf("\n");

	finishCK = omp_get_wtime();
	double duration = (double)(finishCK - startCK);

	if (duration > 1) {
		printf("It took me clicks (%f s).\n\n", duration);
	} else {
		printf("It took me clicks (%f ms).\n\n", duration * 1000);
	}
    ////////////////////////////////////////////////
    cvSaveImage("test.png",OutImage);
    cvShowImage("Result",OutImage);
    //cvShowImage("Result",Image2);
    cvWaitKey(0);

    system( "read -n 1 -s -p \"Press any key to continue...\"; echo" );
	return 0;
}
