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

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef DEGREEOF
#define DEGREEOF(a) ((a*180.0)/M_PI)
#endif

#ifndef RADIANOF
#define RADIANOF(a) ((a*M_PI)/180.0)
#endif

using namespace cv;
using namespace std;

//CL載入Program
cl_program load_program(cl_context context, const char* filename) {
    ifstream in(filename, ios_base::binary);
    if(!in.good()) {
        printf("err0\n");
        return 0;
    }

    // get file length
    in.seekg(0, ios_base::end);
    size_t length = in.tellg();
    in.seekg(0, ios_base::beg);

    // read program source
    vector<char> data(length + 1);
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

//二維轉一維(有平行)
void parallel_in2_out1(int arrayXY, int dirTmp, float modAngle, float* dataIn, float** out1_array){
	/*
    tmpConst[]
    0   tmpSin
    1   tmpCos
    2   arrayXY
    3   lineCount
    4   lineHalf
    */
    float tmpAngle = (float)dirTmp*modAngle;         //角度
    float tmpRadian = RADIANOF(tmpAngle);  //弧度
    int tmpRange = (int)(tmpAngle/45);             //分成四象限，現在四象限作法相同，之後，後續可針對各象限做優化處理
    float tmpSin = sin(tmpRadian);
    float tmpCos = cos(tmpRadian);

    //int dir = 180/modAngle;
    int lineCount = arrayXY;
    int lineHalf = arrayXY/2;

    float tmpConst[5]={tmpSin,tmpCos,(float)arrayXY,(float)lineCount,(float)lineHalf};
    float* tmpConstIndex = (float *)tmpConst;
    float tmp_out1_array[lineCount*4];
    for (int i=0;i<lineCount;i++){
        for (int j=0;j<4;j++){
            tmp_out1_array[i*4+j]=out1_array[i][j];
        }
    }
    float* tmp_out1_arrayIndex = (float *)tmp_out1_array;
    float tmp_dataIn[arrayXY*arrayXY];
    for (int i=0;i<arrayXY*arrayXY;i++){
        tmp_dataIn[i]=dataIn[i];
    }
    float* tmp_dataInIndex = (float *)tmp_dataIn;

    /////////////////////CL 配置
	cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        cerr << "Unable to get platforms\n";
        return;
    }

    vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        cerr << "Unable to get platform ID\n";
        return;
    }

    /////////////////////CL Device
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
    if(context == 0) {
        cerr << "Can't create OpenCL context\n";
        return;
    }

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    //cout << "Device: " << devname.c_str() << "\n";

    /////////////////////CL queue
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return;
    }

    /////////////////////CL buffer
    cl_mem cl_const = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*5, &tmpConstIndex[0], NULL);
    cl_mem cl_data_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*arrayXY*arrayXY, &tmp_dataInIndex[0], NULL);
    cl_mem cl_out1_array = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*lineCount*4, NULL, NULL);//out1_array

    if(cl_const == 0 || cl_data_in == 0 || cl_out1_array == 0) {
        cerr << "Can't create OpenCL buffer\n";
        clReleaseMemObject(cl_const);
        clReleaseMemObject(cl_data_in);
        clReleaseMemObject(cl_out1_array);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    /////////////////////CL program
    cl_program program = load_program(context, "shader1.cl");
    if(program == 0) {
        cerr << "Can't load or build program\n";
        clReleaseMemObject(cl_const);
        clReleaseMemObject(cl_data_in);
        clReleaseMemObject(cl_out1_array);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    /////////////////////CL kernel
    cl_kernel adder = clCreateKernel(program, "adder", 0);
    if(adder == 0) {
        cerr << "Can't load kernel\n";
        clReleaseProgram(program);
        clReleaseMemObject(cl_const);
        clReleaseMemObject(cl_data_in);
        clReleaseMemObject(cl_out1_array);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    /////////////////////CL 資料傳入
    clSetKernelArg(adder, 0, sizeof(cl_const), &cl_const);
    clSetKernelArg(adder, 1, sizeof(cl_data_in), &cl_data_in);
    clSetKernelArg(adder, 2, sizeof(cl_out1_array), &cl_out1_array);

    size_t work_size = lineCount;

    /////////////////////CL 運算出來
    err = clEnqueueNDRangeKernel(queue, adder, 1, 0, &work_size, 0, 0, 0, 0);

    if(err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, cl_out1_array, CL_TRUE, 0, sizeof(float)*lineCount*4, &tmp_out1_arrayIndex[0], 0, 0, 0);

        for (int i=0;i<lineCount;i++){
            for (int j=0;j<4;j++){
                out1_array[i][j]=tmp_out1_arrayIndex[i*4+j];
            }
        }
        
    }

    /////////////////////CL 釋放
    clReleaseKernel(adder);
    clReleaseProgram(program);
    clReleaseMemObject(cl_const);
    clReleaseMemObject(cl_data_in);
    clReleaseMemObject(cl_out1_array);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

//二維轉一維(無平行)
void in2_out1(int arrayXY, int dirTmp, float modAngle, float* dataIn, float** out1_array){
    float tmpAngle = (float)dirTmp*modAngle;         //角度
    float tmpRadian = RADIANOF(tmpAngle);  //弧度
    int tmpRange = (int)(tmpAngle/45);             //分成四象限，現在四象限作法相同，之後，後續可針對各象限做優化處理
    float tmpSin = sin(tmpRadian);
    float tmpCos = cos(tmpRadian);

    //int dir = 180/modAngle;
    int lineCount = arrayXY;
    int lineHalf = arrayXY/2;

    for (int lineNumber=0;lineNumber<lineCount;lineNumber++){
        out1_array[lineNumber][3]=tmpCos;
        out1_array[lineNumber][2]=tmpSin;

        float tmpLineCount = 0;    //該線經過幾格的格數
        float tmpLineAdd = 0;      //該線經過之地的加總數值
        for (int y=0;y<arrayXY;y++){
            for (int x=0;x<arrayXY;x++){
                //用距離公式判斷是否該線有經過該XY
                float tmpDis = abs(tmpSin*(float)(x-lineHalf)-tmpCos*(float)(y-lineHalf)-(lineNumber-lineHalf));
                if (tmpDis<0.5){
                    tmpLineCount++;
                    tmpLineAdd+=dataIn[y*arrayXY+x];
                }
            }
        }

        out1_array[lineNumber][0]=tmpLineAdd;
        out1_array[lineNumber][1]=tmpLineCount;
    }

    /*if (tmpRange==0 || tmpRange==3){
        //(sx-r-0.5)/c < y < (sx-r+0.5)/c
        for (int lineNumber=0;lineNumber<lineCount;lineNumber++){
            out1_array[lineNumber][3]=tmpCos;
            out1_array[lineNumber][2]=tmpSin;

            double tmpLineCount = 0;    //該線經過幾格的格數
            double tmpLineAdd = 0;      //該線經過之地的加總數值
            for (int y=0;y<arrayXY;y++){
                for (int x=0;x<arrayXY;x++){
                    //用距離公式判斷是否該線有經過該XY
                    double tmpDis = abs(tmpSin*(double)(x-lineHalf)-tmpCos*(double)(y-lineHalf)-(lineNumber-lineHalf));
                    if (tmpDis<0.5){
                        tmpLineCount++;
                        tmpLineAdd+=dataIn[y*arrayXY+x];
                    }
                }
            }
            out1_array[lineNumber][0]=tmpLineAdd;
            out1_array[lineNumber][1]=tmpLineCount;
        }

    } else if (tmpRange==1 || tmpRange==2) {
        //(cy+r-0.5)/s < x < (cy+r+0.5)/s
        for (int lineNumber=0;lineNumber<lineCount;lineNumber++){
            out1_array[lineNumber][2]=tmpSin;
            out1_array[lineNumber][3]=tmpCos;

            double tmpLineCount = 0;    //該線經過幾格的格數
            double tmpLineAdd = 0;      //該線經過之地的加總數值
            for (int y=0;y<arrayXY;y++){
                for (int x=0;x<arrayXY;x++){
                    //用距離公式判斷是否該線有經過該XY
                    double tmpDis = abs(tmpSin*(double)(x-lineHalf)-tmpCos*(double)(y-lineHalf)-(lineNumber-lineHalf));
                    if (tmpDis<0.5){
                        tmpLineCount++;
                        tmpLineAdd+=dataIn[y*arrayXY+x];
                    }
                }
            }
            out1_array[lineNumber][0]=tmpLineAdd;
            out1_array[lineNumber][1]=tmpLineCount;
        }
    }*/
}

//一維轉二維(有平行)
void parallel_in1_out2(int arrayXY, int dirTmp, float modAngle, float** out1_array, float** out1_array2, float* out2_array2){
    /*
    tmpConst[]
    0   tmpSin
    1   tmpCos
    2   arrayXY
    3   lineCount
    4   lineHalf
    */

    int lineCount = arrayXY;
    int lineHalf = arrayXY/2;

    //準備傳給CL用的指標
    float tmpConst[5]={out1_array[0][2],out1_array[0][3],(float)arrayXY,(float)lineCount,(float)lineHalf};
    float* tmpConstIndex = (float *)tmpConst;

    float tmp_out1_array[lineCount*4];
    for (int i=0;i<lineCount;i++){
        for (int j=0;j<4;j++){
            tmp_out1_array[i*4+j]=out1_array[i][j];
        }
    }
    float* tmp_out1_arrayIndex = (float *)tmp_out1_array;

    float tmp_out1_array2[lineCount*4];
    for (int i=0;i<lineCount;i++){
        for (int j=0;j<4;j++){
            tmp_out1_array2[i*4+j]=out1_array2[i][j];
        }
    }
    float* tmp_out1_array2Index = (float *)tmp_out1_array2;

    float tmp_out2_array2[arrayXY*arrayXY];
    for (int j=0;j<arrayXY*arrayXY;j++){
        tmp_out2_array2[j]=out2_array2[j];
    }
    float* tmp_out2_array2Index = (float *)tmp_out2_array2;

    /////////////////////CL 配置
	cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        cerr << "Unable to get platforms\n";
        return;
    }

    vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        cerr << "Unable to get platform ID\n";
        return;
    }

    /////////////////////CL Device
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
    if(context == 0) {
        cerr << "Can't create OpenCL context\n";
        return;
    }

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    //cout << "Device: " << devname.c_str() << "\n";

    /////////////////////CL queue
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return;
    }

    /////////////////////CL buffer
    cl_mem cl_const = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*5, &tmpConstIndex[0], NULL);
    cl_mem cl_out1_array = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*lineCount*4, &tmp_out1_arrayIndex[0], NULL);
    cl_mem cl_out1_array2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*lineCount*4, &tmp_out1_array2Index[0], NULL);
    cl_mem cl_out2_array2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*arrayXY*arrayXY, NULL, NULL);

    if(cl_const == 0 || cl_out1_array == 0 || cl_out1_array2 == 0 || cl_out2_array2 == 0) {
        cerr << "Can't create OpenCL buffer\n";
        clReleaseMemObject(cl_const);
        clReleaseMemObject(cl_out1_array);
        clReleaseMemObject(cl_out1_array2);
        clReleaseMemObject(cl_out2_array2);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    /////////////////////CL program
    cl_program program = load_program(context, "shader3.cl");
    if(program == 0) {
        cerr << "Can't load or build program\n";
        clReleaseMemObject(cl_const);
        clReleaseMemObject(cl_out1_array);
        clReleaseMemObject(cl_out1_array2);
        clReleaseMemObject(cl_out2_array2);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    /////////////////////CL kernel
    cl_kernel adder = clCreateKernel(program, "adder", 0);
    if(adder == 0) {
        cerr << "Can't load kernel\n";
        clReleaseProgram(program);
        clReleaseMemObject(cl_const);
        clReleaseMemObject(cl_out1_array);
        clReleaseMemObject(cl_out1_array2);
        clReleaseMemObject(cl_out2_array2);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    /////////////////////CL 資料傳入
    clSetKernelArg(adder, 0, sizeof(cl_const), &cl_const);
    clSetKernelArg(adder, 1, sizeof(cl_out1_array), &cl_out1_array);
    clSetKernelArg(adder, 2, sizeof(cl_out1_array2), &cl_out1_array2);
    clSetKernelArg(adder, 3, sizeof(cl_out2_array2), &cl_out2_array2);

    int dataSize = arrayXY*arrayXY;
    size_t work_size = dataSize;

    /////////////////////CL 運算出來
    err = clEnqueueNDRangeKernel(queue, adder, 1, 0, &work_size, 0, 0, 0, 0);

    if(err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, cl_out2_array2, CL_TRUE, 0, sizeof(float)*arrayXY*arrayXY, &tmp_out2_array2Index[0], 0, 0, 0);

        for (int j=0;j<arrayXY*arrayXY;j++){
            out2_array2[j]+=tmp_out2_array2[j];
        }
    }

    /////////////////////CL 釋放
    clReleaseKernel(adder);
    clReleaseProgram(program);
    clReleaseMemObject(cl_const);
    clReleaseMemObject(cl_out1_array);
    clReleaseMemObject(cl_out1_array2);
    clReleaseMemObject(cl_out2_array2);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

//一維轉二維(無平行)
void in1_out2(int arrayXY, int dirTmp, float modAngle, float** out1_array, float** out1_array2, float* out2_array2){
    //int tmpAngle = dirTmp*modAngle;         //角度
    //double tmpRadian = RADIANOF(tmpAngle);  //弧度
    //int tmpRange = tmpAngle/45;             //分成四象限，現在四象限作法相同，之後，後續可針對各象限做優化處理
    //double tmpSin = sin(tmpRadian);
    //double tmpCos = cos(tmpRadian);

    //int dir = 180/modAngle;
    int lineCount = arrayXY;
    int lineHalf = arrayXY/2;

    //ㄧ維轉二維
    for (int y=0;y<arrayXY;y++){
        for (int x=0;x<arrayXY;x++){
            float tmpX = (float)(x-lineHalf);
	        float tmpY = (float)(y-lineHalf);

            for (int lineNumber=0;lineNumber<lineCount;lineNumber++){
                float tmpSin = out1_array[lineNumber][2];
                float tmpCos = out1_array[lineNumber][3];

                float tmp2_1 = out1_array[lineNumber][0]-out1_array2[lineNumber][0];
                float tmp2_2 = out1_array[lineNumber][1];
                float tmp2=tmp2_1/tmp2_2;
                //用距離公式判斷是否該線有經過該XY
                float tmpDis = abs(tmpSin*tmpX-tmpCos*tmpY-(lineNumber-lineHalf));
                if (tmpDis<0.5){
                    out2_array2[y*arrayXY+x]+=tmp2;
                }
            }
        }
    }
}

//主要運算，從二維轉一維的資料準備，到一維轉二維的實質運算
void mainRun(int needParallel, float modAngle, int loop, int arrayXY, uchar* dataInOrs, uchar* dataOut){
    cout << "~~ 前置作業 二維轉一維 ~~" << endl;

    float dataInTmp[arrayXY*arrayXY];
    for (int ii=0;ii<arrayXY*arrayXY;ii++){
        dataInTmp[ii]=(float)dataInOrs[ii];
    }
    float* dataIn = (float *)dataInTmp;
    /////////////////////前置作業 二維轉一維 變數
    int dir = (int)(180/modAngle);
    int lineCount = arrayXY;
    int lineHalf = arrayXY/2;
    double out1_array[dir*2][lineCount][4];
    for (int i=0;i<dir*2;i++){
        for (int j=0;j<lineCount;j++){
            for (int k=0;k<4;k++){
                out1_array[i][j][k]=0;
            }
        }
    }
    //y=tan()*x-lineNumber*cos()
    //d(P,L) => -0.5<sin*X-cos*Y-lineNumber<0.5
    //0     加總值
    //1     count
    //2     sin()
    //3     cos()

    /////////////////////前置作業 ㄧ維轉二維 變數
    float out1_array2[lineCount][4];

    float out2_array2[arrayXY*arrayXY];
    for (int ii=0;ii<arrayXY*arrayXY;ii++) out2_array2[ii]=0;

    /////////////////////實際算法 二維轉一維
    for (int dirTmp=0;dirTmp<dir;dirTmp++){
        float tmp_out1_array[lineCount][4];
        for (int j=0;j<lineCount;j++){
            for (int k=0;k<4;k++){
                tmp_out1_array[j][k]=0;
            }
        }

        float **ptrOut = (float**)malloc(sizeof(double) * lineCount);
        for (int i=0;i<lineCount;i++){
            ptrOut[i] = (float*)malloc(sizeof(double) * 4);
        }
        for(int i=0;i<lineCount;i++) for(int j=0;j<4;j++) ptrOut[i][j] = 0;


        //二維轉一維 只算180度內
        if (needParallel==0) {
            in2_out1(arrayXY,dirTmp,modAngle,dataIn,ptrOut);
        } else if (needParallel==1) {
            parallel_in2_out1(arrayXY,dirTmp,modAngle,dataIn,ptrOut);
        } else {
            return;
        }

        for (int j=0;j<lineCount;j++){
            for (int k=0;k<4;k++){
                out1_array[dirTmp][j][k]=ptrOut[j][k];
            }
        }
        for (int i=0;i<lineCount;i++){
            free(ptrOut[i]);
        }
        free(ptrOut);

        //二維轉一維 算180度到360度
        for (int lineNumber=0;lineNumber<lineCount;lineNumber++){
            out1_array[dirTmp+dir][lineCount-1-lineNumber][2]=-out1_array[dirTmp][lineNumber][2];
            out1_array[dirTmp+dir][lineCount-1-lineNumber][3]=-out1_array[dirTmp][lineNumber][3];

            out1_array[dirTmp+dir][lineCount-1-lineNumber][0]=out1_array[dirTmp][lineNumber][0];
            out1_array[dirTmp+dir][lineCount-1-lineNumber][1]=out1_array[dirTmp][lineNumber][1];
        }
    }

    cout << "~~ 開始 BackProject ㄧ維轉二維 ~~" << endl;
    /////////////////////實際算法 ㄧ維轉二維
    for (int loopIndex=0;loopIndex<loop;loopIndex++){
        for (int dirTmp=0;dirTmp<dir*2;dirTmp++){
            float tmpAngle = (float)dirTmp*modAngle;
            float tmpRadian = RADIANOF(tmpAngle);
            int tmpRange = tmpAngle/45;
            float tmpSin = sin(tmpRadian);
            float tmpCos = cos(tmpRadian);

            float **ptrOut = (float**)malloc(sizeof(double) * lineCount);
            for (int i=0;i<lineCount;i++){
                ptrOut[i] = (float*)malloc(sizeof(double) * 4);
            }
            for(int i=0;i<lineCount;i++) for(int j=0;j<4;j++) ptrOut[i][j] = out1_array2[i][j];

            //前置作業 上一次的結果，再二維轉一維，待會相減平均用
            //in2_out1(arrayXY,dirTmp,modAngle,out2_array2,ptrOut);
            parallel_in2_out1(arrayXY,dirTmp,modAngle,out2_array2,ptrOut);
            /*if (needParallel==0) {
                in2_out1(arrayXY,dirTmp,modAngle,out2_array2,ptrOut);
            } else if (needParallel==1) {
                parallel_in2_out1(arrayXY,dirTmp,modAngle,out2_array2,ptrOut);
            } else {
                return;
            }*/

            for (int j=0;j<lineCount;j++){
                for (int k=0;k<4;k++){
                    out1_array2[j][k]=ptrOut[j][k];
                }
            }

            for (int i=0;i<lineCount;i++){
                free(ptrOut[i]);
            }
            free(ptrOut);

            //ㄧ維轉二維
            float** ptr_out1_array2 = (float**)malloc(sizeof(double)*lineCount);
            for (int i=0;i<lineCount;i++){
                ptr_out1_array2[i] = (float*)malloc(sizeof(double)*4);
            }
            for(int i=0;i<lineCount;i++) for(int j=0;j<4;j++) ptr_out1_array2[i][j] = out1_array2[i][j];

            float* ptr_out2_array2 = (float*)out2_array2;

            float** ptr_out1_array = (float**)malloc(sizeof(double)*lineCount);
            for (int i=0;i<lineCount;i++){
                ptr_out1_array[i] = (float*)malloc(sizeof(double)*4);
            }
            for(int i=0;i<lineCount;i++) for(int j=0;j<4;j++) ptr_out1_array[i][j] = out1_array[dirTmp][i][j];

            if (needParallel==0) {
                in1_out2(arrayXY,dirTmp,modAngle, ptr_out1_array, ptr_out1_array2, ptr_out2_array2);
            } else if (needParallel==1) {
                parallel_in1_out2(arrayXY,dirTmp,modAngle, ptr_out1_array, ptr_out1_array2, ptr_out2_array2);
            } else {
                return;
            }

            for (int i=0;i<lineCount;i++){
                free(ptr_out1_array2[i]);
                free(ptr_out1_array[i]);
            }
            free(ptr_out1_array2);
            free(ptr_out1_array);

        }
        for (int ii=0;ii<arrayXY*arrayXY;ii++){
            if (out2_array2[ii]>255) {
                out2_array2[ii] = 255;
            } else if (out2_array2[ii]<0) {
                out2_array2[ii] = 0;
            }
        }
    }

    /////////////////////結束 輸出資料
    for (int ii=0;ii<arrayXY*arrayXY;ii++){
        if (out2_array2[ii]>255) {
            dataOut[ii] = (uchar)255;
        } else if (out2_array2[ii]<0) {
            dataOut[ii] = (uchar)0;
        } else {
            dataOut[ii] = (uchar)out2_array2[ii];
        }
    }
}

//主要運算的Print提示
void mainVoid(int needParallel, int indexRGB, float modAngle, int loop, int arrayXY, uchar* pdataIn, uchar* pdataOut){
    if (needParallel==0){
        if (indexRGB==0) cout << "\n~~ 運算 B （無平行)  ~~" << endl;
        else if (indexRGB==1) cout << "\n~~ 運算 G （無平行)  ~~" << endl;
        else if (indexRGB==2) cout << "\n~~ 運算 R （無平行)  ~~" << endl;
    } else if (needParallel==1){
        if (indexRGB==0) cout << "\n~~ 運算 B （有平行) ~~" << endl;
        else if (indexRGB==1) cout << "\n~~ 運算 G （有平行) ~~" << endl;
        else if (indexRGB==2) cout << "\n~~ 運算 R （有平行) ~~" << endl;
    }
    mainRun(needParallel,modAngle,loop,arrayXY,pdataIn,pdataOut);
}

//時間輸出
void timePrint(double duration){
    if (duration > 60) {
        int tmpMin = (int)duration/60;
        int tmpSec = (int)duration%60;
        cout << "It took me clicks (" << tmpMin << "min, " << tmpSec << "s)." << endl;
    } else if (duration > 1) {
        cout << "It took me clicks (" << duration << "s)." << endl;
    } else {
        cout << "It took me clicks (" << duration*1000 << "ms)." << endl;
    }
}

//時間的完整輸出
void timePrintAll(int needParallel, double duration, double duration0, double duration1){
    cout << "\n#####################" << endl;
    cout << "\n完整程式結束，所花時間 : " << endl;
    timePrint(duration);

    if (needParallel==0){
        cout << "\n無平行化程式，所花時間 : " << endl;
        timePrint(duration0);

    } else if (needParallel==1){
        cout << "\n平行化程式，所花時間 : " << endl;
        timePrint(duration1);

    } else if (needParallel==2){
        cout << "\n平行化程式，所花時間 : " << endl;
        timePrint(duration1);

        cout << "\n無平行化程式，所花時間 : " << endl;
        timePrint(duration0);

        cout << "\n平行化快了 : " << duration0/duration1 << "倍" << endl;
    }
}

int main(int argc, const char * argv[]) {

    cout << "\n##程式開始##" << endl;

    /////////////////////前置作業 資料準備
    float modAngle = 45;
    int loop = 4;
    int imageNumber = 1;
    int needParallel=0;

    /////////////////////接收資料
    cout << "輸入圖片編號(ex: 1 or 3)：";
    cin >> imageNumber;
    string imageNumberString="test";
    string tmpString;
    stringstream ss(tmpString);
    ss << imageNumber;
    imageNumberString = imageNumberString + ss.str();
    imageNumberString = imageNumberString + ".png";
    /*while(1){
        cout << "輸入modAngle(請輸入180度能被整除的正整數)：";
        cin >> modAngle;
        if (360%modAngle==0 && modAngle%1==0) break;
    }*/
    cout << "輸入modAngle(請輸入180度能被整除的值) :";
    cin >> modAngle;
    /*while(1){
        cout << "輸入loop(大於0的正整數)：";
        cin >> loop;
        if (loop>=0 && loop%1==0) break;
    }*/
    cout << "輸入loop :";
    cin >> loop;

    cout << "輸入是否需要平行(0為否，1為是，2為兩種都要) :";
    cin >> needParallel;

    cout << "\n##輸入資料確認##" << endl;
    cout << "圖片      : " << imageNumberString << endl;
    cout << "modAngle : " << modAngle << endl;
    cout << "loop     : " << loop << endl;
    cout << "needParallel : " << needParallel << endl;

    /////////////////////前置作業 計時開始
	double omp_get_wtime(void);
	double startCK=0, finishCK=0;
    double startCK0=0, finishCK0=0, startCK1=0, finishCK1=0;
    double duration = 0;
    double duration0 = 0;
    double duration1 = 0;
	startCK = omp_get_wtime();

    /////////////////////前置作業 CV 初始化
    //讀檔
    string file = "finalImage/"+imageNumberString;
    cout << "\n##圖片資料讀取##" << endl;
    cout << "File   :" << file << endl;
    Mat inImageOrs = imread(file,1);
    Mat inImage = imread(file,1);
    Mat outImage = imread(file,1);
    
    //修正Size 改為奇數
    if (inImageOrs.cols%2==0){
        int sizeXY = inImageOrs.cols+11;
        resize(inImageOrs, inImage, Size(sizeXY, sizeXY));
        resize(inImageOrs, outImage, Size(sizeXY, sizeXY));

        uchar* dataOrs = (uchar *)inImageOrs.data;
        uchar* dataIn = (uchar *)inImage.data;

        for (int y=0;y<sizeXY;y++){
            for (int x=0;x<sizeXY;x++){
                int tmpId = (y*sizeXY+x)*3;
                int tmpId2 = ((y-5)*sizeXY+(x-5))*3;
                if (x<5 || x>sizeXY-7 || y<5 || y>sizeXY-7){
                    /*dataIn[tmpId] = 0;
                    dataIn[tmpId+1] = 0;
                    dataIn[tmpId+2] = 0;*/
                    inImage.at<Vec3b>(x,y)[0] = 255;
                    inImage.at<Vec3b>(x,y)[1] = 255;
                    inImage.at<Vec3b>(x,y)[2] = 255;
                } else {
                    /*dataIn[tmpId] = dataOrs[tmpId2];
                    dataIn[tmpId+1] = dataOrs[tmpId2+1];
                    dataIn[tmpId+2] = dataOrs[tmpId2+2];*/
                    inImage.at<Vec3b>(x,y)[0] = inImageOrs.at<Vec3b>((x-5),(y-5))[0];
                    inImage.at<Vec3b>(x,y)[1] = inImageOrs.at<Vec3b>((x-5),(y-5))[1];
                    inImage.at<Vec3b>(x,y)[2] = inImageOrs.at<Vec3b>((x-5),(y-5))[2];
                }
            }
        }
    } else {
        int sizeXY = inImageOrs.cols+10;
        resize(inImageOrs, inImage, Size(sizeXY, sizeXY));
        resize(inImageOrs, outImage, Size(sizeXY, sizeXY));

        uchar* dataOrs = (uchar *)inImageOrs.data;
        uchar* dataIn = (uchar *)inImage.data;

        for (int y=0;y<sizeXY;y++){
            for (int x=0;x<sizeXY;x++){
                int tmpId = (y*sizeXY+x)*3;
                int tmpId2 = ((y-5)*sizeXY+(x-5))*3;
                if (x<5 || x>sizeXY-6 || y<5 || y>sizeXY-6){
                    /*dataIn[tmpId] = 0;
                    dataIn[tmpId+1] = 0;
                    dataIn[tmpId+2] = 0;*/
                    inImage.at<Vec3b>(x,y)[0] = 255;
                    inImage.at<Vec3b>(x,y)[1] = 255;
                    inImage.at<Vec3b>(x,y)[2] = 255;
                } else {
                    /*dataIn[tmpId] = dataOrs[tmpId2];
                    dataIn[tmpId+1] = dataOrs[tmpId2+1];
                    dataIn[tmpId+2] = dataOrs[tmpId2+2];*/
                    inImage.at<Vec3b>(x,y)[0] = inImageOrs.at<Vec3b>((x-5),(y-5))[0];
                    inImage.at<Vec3b>(x,y)[1] = inImageOrs.at<Vec3b>((x-5),(y-5))[1];
                    inImage.at<Vec3b>(x,y)[2] = inImageOrs.at<Vec3b>((x-5),(y-5))[2];
                }
            }
        }
    }
    //取得Size
    CvSize SizeIn=inImage.size();
    CvSize SizeOut = outImage.size();
    int inSize[2]={SizeIn.height,SizeIn.width};
    int outSize[2]={SizeOut.height,SizeOut.width};
    int* ins = (int *)inSize;
    int* outs = (int *)outSize;

    const int arrayXY = inSize[0];
    //Get RGB指標
    uchar* dataIn = (uchar *)inImage.data;
    uchar* dataOut = (uchar *)outImage.data;
    //Get RGB Data
    uchar dataInR[arrayXY*arrayXY];
    uchar dataOutR[arrayXY*arrayXY];
    uchar dataInG[arrayXY*arrayXY];
    uchar dataOutG[arrayXY*arrayXY];
    uchar dataInB[arrayXY*arrayXY];
    uchar dataOutB[arrayXY*arrayXY];

    for (int y=0;y<arrayXY;y++){
        for (int x=0;x<arrayXY;x++){
            int tmpId = (y*arrayXY+x)*3;
            int tmpId2 = y*arrayXY+x;
            dataInR[tmpId2] = dataIn[tmpId+2];
            dataOutR[tmpId2] = dataOut[tmpId+2];

            dataInG[tmpId2] = dataIn[tmpId+1];
            dataOutG[tmpId2] = dataOut[tmpId+1];

            dataInB[tmpId2] = dataIn[tmpId];
            dataOutB[tmpId2] = dataOut[tmpId];
        }
    }

    //RGB Data的指標
    uchar* pdataInR = (uchar *)dataInR;
    uchar* pdataOutR = (uchar *)dataOutR;
    uchar* pdataInG = (uchar *)dataInG;
    uchar* pdataOutG = (uchar *)dataOutG;
    uchar* pdataInB = (uchar *)dataInB;
    uchar* pdataOutB = (uchar *)dataOutB;

    /////////////////////前置作業 CL 初始化
    const int DATA_SIZE = inSize[0]*inSize[1]*3;
    size_t work_size = inSize[0]*inSize[1];

    cout << "\n##圖片資料 Size 確認##" << endl;
    cout << "height, width  : " << inSize[0] << ", " << inSize[1] << endl;
    cout << "DataSize       : " << DATA_SIZE << endl;
    cout << "WorkSize       : " << work_size << endl;
    cout << "arrayXY        : " << arrayXY << endl;

    /////////////////////開始運算
    if (needParallel==0){
        cout << "\n~~ 開始運算（無平行) ~~" << endl;
        startCK0 = omp_get_wtime();
        mainVoid(0,0,modAngle,loop,arrayXY,pdataInB,pdataOutB);
        mainVoid(0,1,modAngle,loop,arrayXY,pdataInG,pdataOutG);
        mainVoid(0,2,modAngle,loop,arrayXY,pdataInR,pdataOutR);
        finishCK0 = omp_get_wtime();
        cout << "\n~~ 運算完畢（無平行) ~~" << endl;

    } else if (needParallel==1){
        cout << "\n~~ 開始運算（有平行) ~~" << endl;
        startCK1 = omp_get_wtime();
        mainVoid(1,0,modAngle,loop,arrayXY,pdataInB,pdataOutB);
        mainVoid(1,1,modAngle,loop,arrayXY,pdataInG,pdataOutG);
        mainVoid(1,2,modAngle,loop,arrayXY,pdataInR,pdataOutR);
        finishCK1 = omp_get_wtime();
        cout << "\n~~ 運算完畢（有平行) ~~" << endl;

    } else if (needParallel==2){
        cout << "\n~~ 開始運算（有平行) ~~" << endl;
        startCK1 = omp_get_wtime();
        mainVoid(1,0,modAngle,loop,arrayXY,pdataInB,pdataOutB);
        mainVoid(1,1,modAngle,loop,arrayXY,pdataInG,pdataOutG);
        mainVoid(1,2,modAngle,loop,arrayXY,pdataInR,pdataOutR);
        finishCK1 = omp_get_wtime();
        cout << "\n~~ 運算完畢（有平行) ~~" << endl;

        cout << "\n#####################" << endl;

        cout << "\n~~ 開始運算（無平行) ~~" << endl;
        startCK0 = omp_get_wtime();
        mainVoid(0,0,modAngle,loop,arrayXY,pdataInB,pdataOutB);
        mainVoid(0,1,modAngle,loop,arrayXY,pdataInG,pdataOutG);
        mainVoid(0,2,modAngle,loop,arrayXY,pdataInR,pdataOutR);
        finishCK0 = omp_get_wtime();
        cout << "\n~~ 運算完畢（無平行) ~~" << endl;
    }
    
    /////////////////////結束 輸出資料放回
    for (int y=0;y<arrayXY;y++){
        for (int x=0;x<arrayXY;x++){
            int tmpId = (y*arrayXY+x)*3;
            int tmpId2 = y*arrayXY+x;

            dataOut[tmpId+2] = dataOutR[tmpId2];
            dataOut[tmpId+1] = dataOutG[tmpId2];
            dataOut[tmpId] = dataOutB[tmpId2];

            /*dataOut[tmpId+2] = dataOutR[tmpId2];
            dataOut[tmpId+1] = dataOutR[tmpId2];
            dataOut[tmpId] = dataOutR[tmpId2];*/
        }
    }

    /////////////////////收尾 計時結束
    finishCK = omp_get_wtime();
    duration = (double)(finishCK - startCK);
    duration0 = (double)(finishCK0 - startCK0);
    duration1 = (double)(finishCK1 - startCK1);
    timePrintAll(needParallel,duration,duration0,duration1);

    /////////////////////收尾 釋放
    /*free(pdataInR);
    free(pdataOutR);
    free(pdataInG);
    free(pdataOutG);
    free(pdataInB);
    free(pdataOutB);*/

    /////////////////////CV show & save image
    //imwrite("tmp.png",outImage);
    system( "read -n 1 -s -p \"Press any key to continue...\"; echo" );

    Mat img_outShow(arrayXY, arrayXY*2, inImage.type() );
    Mat part;
    part = img_outShow(Rect(0,0,arrayXY,arrayXY)); 
    inImage.copyTo(part);
    part =  img_outShow(Rect(arrayXY,0,arrayXY,arrayXY)); 
    outImage.copyTo(part);

    cvNamedWindow("out",CV_WINDOW_NORMAL);
    imshow("out",img_outShow);
    cvWaitKey(0);

	return 0;
}
