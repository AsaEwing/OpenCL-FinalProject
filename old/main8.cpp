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

using namespace cv;
using namespace std;

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

void print_Out1(int count, int indexArg, double* outArray){
    cout << "inArray " << indexArg << " :";
    for (int ii=0;ii<count;ii++){
        cout << right << setw(4) << outArray[ii];
        if (ii!=count-1) cout << ",";
    }
    cout << "\n";
}

void print_Out2(int sizeXY,int count, int indexArg, uchar* orsArray, double* outArray){
    cout << "outArray " << indexArg << ":\n" ;
    double err = 0;
    double maxErr = 0;
    double minErr = 0;
    double tmp[sizeXY*sizeXY];
    for (int y=0;y<sizeXY;y++){
        for (int x=0;x<sizeXY;x++){
            //cout << right << setw(10) << outArray[y][x];
            //if (x!=sizeXY-1) cout << ",";
            int tmpId = y*sizeXY+x;
            tmp[tmpId]=outArray[tmpId]-orsArray[tmpId];
            if (tmp[tmpId]>maxErr) maxErr=tmp[tmpId];
            if (tmp[tmpId]<minErr) minErr=tmp[tmpId];
            err+=pow(tmp[tmpId],2);
        }
        //cout << endl;
    }

    /*cout << "\n";
    for (int y=0;y<sizeXY;y++){
        for (int x=0;x<sizeXY;x++){
            cout << right << setw(10) << tmp[y][x];
            if (x!=sizeXY-1) cout << ",";
        }
        cout << endl;
    }*/

    cout << "Err all   : " << err << "\n";
    cout << "Err mean 2: " << err/sizeXY/sizeXY << "\n";
    cout << "Err mean 1: " << pow(err/sizeXY/sizeXY,0.5) << "\n";
    cout << "Err min   : " << minErr << "\n";
    cout << "Err max   : " << maxErr << "\n\n";

    /*if (indexArg==3){
        cout << "Ors :\n";
        for (int y=0;y<sizeXY;y++){
            for (int x=0;x<sizeXY;x++){
                cout << right << setw(10) << (int)orsArray[y][x];
                if (x!=sizeXY-1) cout << ",";
            }
            cout << endl;
        }
    }*/

    if (indexArg==7){
        double err2 = 0;
        double maxErr2 = 0;
        double minErr2 = 0;
        double tmp2[sizeXY*sizeXY];

        double tmpOut[sizeXY*sizeXY];
        double maxOut = 0;
        double minOut = 0;
        for (int y=0;y<sizeXY;y++){
            for (int x=0;x<sizeXY;x++){
                int tmpId = y*sizeXY+x;
                if (outArray[tmpId]>maxErr2) maxOut=outArray[tmpId];
                if (outArray[tmpId]<minErr2) minOut=outArray[tmpId];
            }
        }

        for (int y=0;y<sizeXY;y++){
            for (int x=0;x<sizeXY;x++){
                int tmpId = y*sizeXY+x;
                if (outArray[tmpId]<0){
                    tmpOut[tmpId]=0;
                } else if (outArray[tmpId]>28){
                    tmpOut[tmpId]=28;
                } else {
                    tmpOut[tmpId]=outArray[tmpId];
                }
                //tmpOut[y][x]=(outArray[y][x]-minOut)/(maxOut-minOut)*28;
            }
        }

        for (int y=0;y<sizeXY;y++){
            for (int x=0;x<sizeXY;x++){
                int tmpId = y*sizeXY+x;
                //cout << right << setw(10) << outArray[y][x];
                //if (x!=sizeXY-1) cout << ",";

                tmp2[tmpId]=tmpOut[tmpId]-orsArray[tmpId];
                if (tmp2[tmpId]>maxErr2) maxErr2=tmp2[tmpId];
                if (tmp2[tmpId]<minErr2) minErr2=tmp2[tmpId];
                err2+=pow(tmp2[tmpId],2);
            }
            //cout << endl;
        }

        cout << "Final :\n";
        cout << "Err all   : " << err2 << "\n";
        cout << "Err mean 2: " << err2/sizeXY/sizeXY << "\n";
        cout << "Err mean 1: " << pow(err2/sizeXY/sizeXY,0.5) << "\n";
        cout << "Err min   : " << minErr2 << "\n";
        cout << "Err max   : " << maxErr2 << "\n\n";
    }
}

void in2_out1(int sizeXY, int indexArg, uchar* inArray ,double* outArray){
    if (indexArg==0){
        int count = sizeXY;
        for (int y=0;y<count;y++){
            outArray[y]=0;
            for (int x=0;x<count;x++){
                outArray[y]+=inArray[y*sizeXY+x];
            }
        }
        print_Out1(count,indexArg,outArray);

    } else if (indexArg==1){
        int count = sizeXY*2-1;
        for (int ii=0;ii<count;ii++){
            outArray[ii]=0;
            for (int x=0;x<sizeXY;x++){
                for (int y=0;y<sizeXY;y++){
                    if (x-y+sizeXY-1==ii){
                        outArray[ii]+=inArray[y*sizeXY+x];
                    }
                }
            }
        }
        print_Out1(count,indexArg,outArray);

    } else if (indexArg==2){
        int count = sizeXY;
        for (int y=0;y<count;y++){
            outArray[y]=0;
            for (int x=0;x<count;x++){
                outArray[y]+=inArray[(sizeXY-1-y)*sizeXY+x];
            }
        }
        print_Out1(count,indexArg,outArray);

    } else if (indexArg==3){
        int count = sizeXY*2-1;
        for (int ii=0;ii<count;ii++){
            outArray[ii]=0;
            for (int x=0;x<sizeXY;x++){
                for (int y=0;y<sizeXY;y++){
                    if (count-x-y-1==ii){
                        outArray[ii]+=inArray[y*sizeXY+x];
                    }
                }
            }
        }
        print_Out1(count,indexArg,outArray);
        cout << "\n";
    }
}

void in1_out2(int sizeXY, int indexArg, uchar* orsArray, double* inArray, double* outArray){
    if (indexArg==0 || indexArg==2){
        int count = sizeXY;
        double tmp[count];

        for (int ii=0;ii<sizeXY;ii++){
            tmp[ii]=0;
            for (int jj=0;jj<sizeXY;jj++){
                if (indexArg==0) tmp[ii]+=outArray[jj*sizeXY+ii];
                else if (indexArg==2) tmp[ii]+=outArray[(count-1-ii)*sizeXY+jj];
            }
        }

        for (int ii=0;ii<count;ii++){
            double tmp2=((double)inArray[ii]-tmp[ii])/(double)sizeXY;
            //cout << "tmp2  "<<tmp2<<"  "<< inArray[ii] <<"  "<< tmp[ii] <<"  "<<"\n";
            for (int x=0;x<sizeXY;x++){
                for (int y=0;y<sizeXY;y++){
                    if (indexArg==0 && x==ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    } else if (indexArg==2 && y==count-1-ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    }
                }
            }
        }
        print_Out2(sizeXY,count,indexArg,orsArray,outArray);

    } else if (indexArg==1 || indexArg==3){
        int count = sizeXY*2-1;
        double tmp[count];

        for (int ii=0;ii<count;ii++){
            tmp[ii]=0;
            for (int y=0;y<sizeXY;y++){
                for (int x=0;x<sizeXY;x++){
                    if (indexArg==1 && x-y+sizeXY-1==ii){
                        tmp[ii]+=outArray[y*sizeXY+x];
                    } else if (indexArg==3 && count-x-y-1==ii){
                        tmp[ii]+=outArray[y*sizeXY+x];
                    }
                }
            }
        }

        for (int ii=0;ii<count;ii++){
            double tmp2 = ii-sizeXY+1;
            if (tmp2<0) tmp2 = -tmp2;
            tmp2 = sizeXY-tmp2;
            tmp2=((double)inArray[ii]-tmp[ii])/tmp2;

            for (int y=0;y<sizeXY;y++){
                for (int x=0;x<sizeXY;x++){
                    if (indexArg==1 && x-y+sizeXY-1==ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    } else if (indexArg==3 && count-x-y-1==ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    }
                }
            }
        }
        print_Out2(sizeXY,count,indexArg,orsArray,outArray);

    } else if (indexArg==4 || indexArg==6){
        int count = sizeXY;
        double tmp[count];

        for (int ii=0;ii<sizeXY;ii++){
            tmp[ii]=0;
            for (int jj=0;jj<sizeXY;jj++){
                if (indexArg==4) tmp[ii]+=outArray[jj*sizeXY+(sizeXY-1-ii)];
                else if (indexArg==6) tmp[ii]+=outArray[ii*sizeXY+jj];
            }
        }

        for (int ii=0;ii<count;ii++){
            double tmp2=((double)inArray[sizeXY-1-ii]-tmp[ii])/(double)sizeXY;
            //cout << "tmp2  "<<tmp2<<"  "<< inArray[ii] <<"  "<< tmp[ii] <<"  "<<"\n";
            for (int x=0;x<sizeXY;x++){
                for (int y=0;y<sizeXY;y++){
                    if (indexArg==4 && x==sizeXY-1-ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    } else if (indexArg==6 && y==ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    }
                }
            }
        }
        print_Out2(sizeXY,count,indexArg,orsArray,outArray);

    } else if (indexArg==5 || indexArg==7){
        int count = sizeXY*2-1;
        double tmp[count];

        for (int ii=0;ii<count;ii++){
            tmp[ii]=0;
            for (int y=0;y<sizeXY;y++){
                for (int x=0;x<sizeXY;x++){
                    if (indexArg==5 && sizeXY-x+y-1==ii){
                        tmp[ii]+=outArray[y*sizeXY+x];
                    } else if (indexArg==7 && x+y==ii){
                        tmp[ii]+=outArray[y*sizeXY+x];
                    }
                }
            }
        }

        for (int ii=0;ii<count;ii++){
            double tmp2 = ii-sizeXY+1;
            if (tmp2<0) tmp2 = -tmp2;
            tmp2 = sizeXY-tmp2;
            tmp2=((double)inArray[count-1-ii]-tmp[ii])/tmp2;

            for (int y=0;y<sizeXY;y++){
                for (int x=0;x<sizeXY;x++){
                    if (indexArg==5 && sizeXY-x+y-1==ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    } else if (indexArg==7 && x+y==ii){
                        outArray[y*sizeXY+x]+=tmp2;
                    }
                }
            }
        }
        print_Out2(sizeXY,count,indexArg,orsArray,outArray);

    }
}

int main(int argc, const char * argv[]) {

    /////////////////////前置作業 計時開始
	double omp_get_wtime(void);
	double startCK, finishCK;
	startCK = omp_get_wtime();

    /////////////////////前置作業 資料準備
    const int arrayXY = 100;
    /*uchar inarray[arrayXY][arrayXY] = {{0,1,2,3},
                                        {0,1,2,3},
                                        {0,1,2,4},
                                        {0,1,2,8}};*/
    /*uchar inarray[arrayXY][arrayXY] = {{0,1,2,3,4},
                                        {0,4,2,3,8},
                                        {255,5,2,4,14},
                                        {0,1,2,0,0},
                                        {0,1,2,0,0}};*/
    /* inarray[arrayXY][arrayXY] = {{0,1,-2,3},
                                    {0,-1,-2,3},
                                    {0,1,2,-4},
                                    {0,-1,-2,8}};*/
    //////inarray[Y][X]
    double to1array0[arrayXY];
    double to1array1[arrayXY*2-1];
    double to1array2[arrayXY];
    double to1array3[arrayXY*2-1];

    /*double tmparray[arrayXY][arrayXY];
    for (int ii=0;ii<arrayXY;ii++){
        for (int jj=0;jj<arrayXY;jj++){
            tmparray[ii][jj]=0;
        }
    }*/

    double tmparray[arrayXY*arrayXY];
    for (int ii=0;ii<arrayXY;ii++){
        for (int jj=0;jj<arrayXY;jj++){
            tmparray[ii*arrayXY+jj]=0;
        }
    }

    /////////////////////CV 初始化
    //int height, width;
	uchar *dataIn;
    uchar *dataOut;

    Mat inImage = imread("finalImage/test1.png",1);
    CvSize SizeIn=inImage.size();

    Mat outImage = imread("finalImage/test1.png",1);
    CvSize SizeOut = outImage.size();
    
    int inSize[2]={SizeIn.height,SizeIn.width};
    int outSize[2]={SizeOut.height,SizeOut.width};
    int *ins = (int *)inSize;
    int *outs = (int *)outSize;

    dataIn = (uchar *)inImage.data;
    dataOut = (uchar *)outImage.data;

    ////////////////////////////////////////////////////////////////////////////////////////////
    /*uchar **ptrIn = (uchar**)malloc(sizeof(uchar) * arrayXY);
    for (int i=0;i<arrayXY;i++){
        ptrIn[i] = (uchar*)malloc(sizeof(uchar) * arrayXY);
    }
    for(int i=0;i<arrayXY;i++) {
        for(int j=0;j<arrayXY;j++) {
            //ptrIn[i][j] = dataIn[i*arrayXY+j];
            ptrIn[i][j] = inImage.data[i*arrayXY+j];
        }
    }*/
    ////////////////////////////////////////////////////////////////////////////////////////////
    
    const int DATA_SIZE = inSize[0]*inSize[1]*3;
    size_t work_size = inSize[0]*inSize[1];

    //uchar data4[DATA_SIZE1];

    cout << "h1, w1: " << inSize[0] << ", " << inSize[1] << "\n";
    cout << "h2, w2: " << outSize[0] << ", " << outSize[1] << "\n";
    cout << "DataSize 1: " << DATA_SIZE << "\n";
    cout << "WorkSize 1: " << work_size << "\n\n";

    cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Result2",CV_WINDOW_AUTOSIZE);

    /////////////////////前置作業 二維資料改二維指標
    /*uchar **ptrIn = (uchar**)malloc(sizeof(uchar) * arrayXY);
    for (int i=0;i<arrayXY;i++){
        ptrIn[i] = (uchar*)malloc(sizeof(uchar) * arrayXY);
    }
    for(int i=0;i<arrayXY;i++) for(int j=0;j<arrayXY;j++) ptrIn[i][j] = inarray[i][j];*/

    /*double **ptrOut = (double**)malloc(sizeof(double) * arrayXY);
    for (int i=0;i<arrayXY;i++){
        ptrOut[i] = (double*)malloc(sizeof(double) * arrayXY);
    }
    for(int i=0;i<arrayXY;i++) for(int j=0;j<arrayXY;j++) ptrOut[i][j] = tmparray[i][j];*/
    double *ptrOut = tmparray;

    /////////////////////前置作業 二維轉一維
    for (int indexArg=0;indexArg<4;indexArg++){
        if (indexArg==0){
            //in2_out1(arrayXY,indexArg,ptrIn,to1array0);
            in2_out1(arrayXY,indexArg,dataIn,to1array0);
        } else if (indexArg==1){
            //in2_out1(arrayXY,indexArg,ptrIn,to1array1);
            in2_out1(arrayXY,indexArg,dataIn,to1array1);
        } else if (indexArg==2){
            //in2_out1(arrayXY,indexArg,ptrIn,to1array2);
            in2_out1(arrayXY,indexArg,dataIn,to1array2);
        } else if (indexArg==3){
            //in2_out1(arrayXY,indexArg,ptrIn,to1array3);
            in2_out1(arrayXY,indexArg,dataIn,to1array3);
        }
    }

    /////////////////////實際算法 ㄧ維轉二維
    int modIndex = 8;
    /*
    Final :
Err all   : 27.9384
Err mean 2: 1.11754
Err mean 1: 1.05714
Err min   : -1.99988
Err max   : 1.99988*/
    for (int indexArg=0;indexArg<modIndex*100;indexArg++){
        int tmpIndex = indexArg%modIndex;
        if (tmpIndex==0){
            //in1_out2(arrayXY,0,ptrIn,to1array0,ptrOut);
            in1_out2(arrayXY,0,dataIn,to1array0,ptrOut);
        } else if (tmpIndex==1){
            //in1_out2(arrayXY,1,ptrIn,to1array1,ptrOut);
            in1_out2(arrayXY,0,dataIn,to1array1,ptrOut);
        } else if (tmpIndex==2){
            //in1_out2(arrayXY,2,ptrIn,to1array2,ptrOut);
            in1_out2(arrayXY,0,dataIn,to1array2,ptrOut);
        } else if (tmpIndex==3){
            //in1_out2(arrayXY,3,ptrIn,to1array3,ptrOut);
            in1_out2(arrayXY,0,dataIn,to1array3,ptrOut);
        } else if (tmpIndex==4){
            //in1_out2(arrayXY,4,ptrIn,to1array0,ptrOut);
            in1_out2(arrayXY,0,dataIn,to1array0,ptrOut);
        } else if (tmpIndex==5){
            //in1_out2(arrayXY,5,ptrIn,to1array1,ptrOut)
            in1_out2(arrayXY,0,dataIn,to1array1,ptrOut);
        } else if (tmpIndex==6){
            //in1_out2(arrayXY,6,ptrIn,to1array2,ptrOut);
            in1_out2(arrayXY,0,dataIn,to1array2,ptrOut);
        } else if (tmpIndex==7){
            //in1_out2(arrayXY,7,ptrIn,to1array3,ptrOut);
            in1_out2(arrayXY,0,dataIn,to1array3,ptrOut);
            for (int y=0;y<arrayXY;y++){
                for (int x=0;x<arrayXY;x++){
                    /*if (ptrOut[y][x]<0){
                        ptrOut[y][x]=0;
                    } else if (ptrOut[y][x]>28){
                        ptrOut[y][x]=28;
                    }*/
                    if (ptrOut[y*arrayXY+x]<0){
                        ptrOut[y*arrayXY+x]=0;
                    } else if (ptrOut[y*arrayXY+x]>225){
                        ptrOut[y*arrayXY+x]=255;
                    }
                }
            }
        }
    }
   
    /////////////////////CL 配置
	/*cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        cerr << "Unable to get platforms\n";
        return 0;
    }

    vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        cerr << "Unable to get platform ID\n";
        return 0;
    }

    /////////////////////CL Device
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
    if(context == 0) {
        cerr << "Can't create OpenCL context\n";
        return 0;
    }

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    cout << "Device: " << devname.c_str() << "\n";

    /////////////////////CL queue
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return 0;
    }

    /////////////////////CL buffer
    cl_mem cl_in1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar)*DATA_SIZE1, &data1[0], NULL);
    cl_mem cl_in2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar)*DATA_SIZE2, &data2[0], NULL);
    cl_mem cl_size1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*2, &ins1[0], NULL);
    cl_mem cl_size2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*2, &ins2[0], NULL);
    cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar)*DATA_SIZE1, NULL, NULL);

    if(cl_in1 == 0 || cl_in2 == 0 || cl_size1 == 0 || cl_size2 == 0 || cl_res == 0) {
        cerr << "Can't create OpenCL buffer\n";
        clReleaseMemObject(cl_in1);
        clReleaseMemObject(cl_in2);
        clReleaseMemObject(cl_size1);
        clReleaseMemObject(cl_size2);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    /////////////////////CL program
    cl_program program = load_program(context, "shader6.cl");
    if(program == 0) {
        cerr << "Can't load or build program\n";
        clReleaseMemObject(cl_in1);
        clReleaseMemObject(cl_in2);
        clReleaseMemObject(cl_size1);
        clReleaseMemObject(cl_size2);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    /////////////////////CL kernel
    cl_kernel adder = clCreateKernel(program, "adder", 0);
    if(adder == 0) {
        cerr << "Can't load kernel\n";
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

    /////////////////////CL 資料傳入
    clSetKernelArg(adder, 0, sizeof(cl_in1), &cl_in1);
    clSetKernelArg(adder, 1, sizeof(cl_in2), &cl_in2);
    clSetKernelArg(adder, 2, sizeof(cl_size1), &cl_size1);
    clSetKernelArg(adder, 3, sizeof(cl_size2), &cl_size2);
    clSetKernelArg(adder, 4, sizeof(cl_res), &cl_res);

    /////////////////////CL 運算出來
    err = clEnqueueNDRangeKernel(queue, adder, 1, 0, &work_size, 0, 0, 0, 0);

    if(err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(uchar) * DATA_SIZE1, &data3[0], 0, 0, 0);
        //&data3[0]
    }

    /////////////////////CL 釋放
    clReleaseKernel(adder);
    clReleaseProgram(program);
    clReleaseMemObject(cl_in1);
    clReleaseMemObject(cl_in2);
    clReleaseMemObject(cl_size1);
    clReleaseMemObject(cl_size2);
    clReleaseMemObject(cl_res);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);*/

    /////////////////////收尾 計時結束
	finishCK = omp_get_wtime();
	double duration = (double)(finishCK - startCK);
    printf("\n");
	if (duration > 1) {
		printf("It took me clicks (%f s).\n\n", duration);
	} else {
		printf("It took me clicks (%f ms).\n\n", duration * 1000);
	}

    /////////////////////收尾 釋放
    /*for (int i=0;i<arrayXY;i++){
        free(ptrIn[i]);
        free(ptrOut[i]);
    }
    free(ptrIn);
    free(ptrOut);*/

    /*for(int i=0;i<arrayXY;i++) {
        for(int j=0;j<arrayXY;j++) {
            if (tmparray[i][j]>255){
                dataOut[i*arrayXY+j] = 255;
            } else if (tmparray[i][j]<0){
                dataOut[i*arrayXY+j] = 0;
            } else {
                dataOut[i*arrayXY+j] = (uchar)tmparray[i][j];
            }
        }
    }*/

    for(int i=0;i<arrayXY*arrayXY;i++) {
            if (tmparray[i]>255){
                dataOut[i] = (uchar)255;
            } else if (tmparray[i]<0){
                dataOut[i] = (uchar)0;
            } else {
                dataOut[i] = (uchar)tmparray[i];
            }
    }

    system( "read -n 1 -s -p \"Press any key to continue...\"; echo" );
    
    /////////////////////CV show image
    imshow("Result",outImage);
    imshow("Result2",inImage);
    cvWaitKey(0);

	return 0;
}
