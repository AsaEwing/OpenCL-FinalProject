__kernel void adder(__global const float* tmpConst, 
					__global const float* dataIn,
					__global float* out1_array)
{	
	int idx = get_global_id(0);
	
	float index = convert_float(idx);
	int lineNumber = idx;
	int lineHalf = tmpConst[4];
	int lineCount = tmpConst[3];

	int arrayXY = tmpConst[2];

	float tmpSin = tmpConst[0];
	float tmpCos = tmpConst[1];

	//int x=idx%tmpConst[2];
	//int y=convert_int((index-convert_float(x))/tmpConst[2]);
	//int lineNumberX = idx%4;
	//int lineNumber = convert_int((index-convert_float(lineNumberX))/lineNumberX);

	out1_array[lineNumber*4+3]=tmpCos;
    out1_array[lineNumber*4+2]=tmpSin;

    float tmpLineCount = 0;
    float tmpLineAdd = 0;
    for (int y=0;y<arrayXY;y++){
        for (int x=0;x<arrayXY;x++){
			float tmpX = convert_float(x-lineHalf);
			float tmpY = convert_float(y-lineHalf);
			float tmpLine = convert_float(lineNumber-lineHalf);
            float tmpDis = tmpSin*tmpX-tmpCos*tmpY-tmpLine;
            if (tmpDis<0.5 && tmpDis>-0.5){
                tmpLineCount=tmpLineCount+1;
                tmpLineAdd=tmpLineAdd+dataIn[y*arrayXY+x];
            }
        }
    }

	out1_array[lineNumber*4+0]=tmpLineAdd;
    out1_array[lineNumber*4+1]=tmpLineCount;
}