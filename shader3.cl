# pragma OPENCL EXTENSION cl_intel_printf :enable

__kernel void adder(__global const float* tmpConst, 
					__global const float* out1_array,
					__global const float* out1_array2,
					__global float* out2_array2)
{	
	int idx = get_global_id(0);
	
	float index = convert_float(idx);
	
	int lineHalf = tmpConst[4];
	int lineCount = tmpConst[3];

	int arrayXY = tmpConst[2];

	float tmpSin = tmpConst[0];
	float tmpCos = tmpConst[1];

	int x=idx%arrayXY;
	int y=convert_int((index-convert_float(x))/convert_float(arrayXY));

	float tmpX = convert_float(x-lineHalf);
	float tmpY = convert_float(y-lineHalf);

    for (int lineNumber=0;lineNumber<lineCount;lineNumber++){
		//float tmpSin = out1_array[lineNumber*4+2];
        //float tmpCos = out1_array[lineNumber*4+3];
		//printf("111");

		float tmp2_1 = convert_float(out1_array[lineNumber*4+0]-out1_array2[lineNumber*4+0]);
    	float tmp2_2 = convert_float(out1_array[lineNumber*4+1]);
    	float tmp2 = convert_float(tmp2_1)/convert_float(tmp2_2);

		float tmpLine = convert_float(lineNumber-lineHalf);
        float tmpDis = tmpSin*tmpX-tmpCos*tmpY-tmpLine;
		/*if (tmp2_1==0){
			out2_array2[y*arrayXY+x]=0;
		} else {
			out2_array2[y*arrayXY+x]=255;
		}*/
		

        if (tmpDis<0.5 && tmpDis>-0.5){
			//out2_array2[y*arrayXY+x]=out2_array2[y*arrayXY+x]+tmp2;
			out2_array2[y*arrayXY+x]+=tmp2;
			//out2_array2[y*arrayXY+x]+=tmpSin*255;
        }
	}

	/*
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
    }*/
    
}