__kernel void adder(__global const float* tmpConst, 
					__global const float* out1_array,
					__global const float* out1_array2,
					__global float* out2_array2)
{	
	int idx = get_global_id(0);
	
	float index = convert_float(idx);
	int lineNumber = idx;
	int lineHalf = tmpConst[4];
	int lineCount = tmpConst[3];

	int arrayXY = tmpConst[2];

	float tmpSin = tmpConst[0];
	float tmpCos = tmpConst[1];
    
    //float tmpSin = out1_array[lineNumber*4+2];
    //float tmpCos = out1_array[lineNumber*4+3];

    float tmp2_1 = out1_array[lineNumber*4+0]-out1_array2[lineNumber*4+0];
    float tmp2_2 = out1_array[lineNumber*4+1];
    float tmp2 = tmp2_1/tmp2_2;

    for (int y=0;y<arrayXY;y++){
        for (int x=0;x<arrayXY;x++){
            float tmpX = convert_float(x-lineHalf);
			float tmpY = convert_float(y-lineHalf);
			float tmpLine = convert_float(lineNumber-lineHalf);
            float tmpDis = tmpSin*tmpX-tmpCos*tmpY-tmpLine;
            if (tmpDis<0.5 && tmpDis>-0.5){
				out2_array2[lineNumber*arrayXY*arrayXY+y*arrayXY+x]=out2_array2[lineNumber*arrayXY*arrayXY+y*arrayXY+x]+tmp2;
            }
    	}
	}
    
}