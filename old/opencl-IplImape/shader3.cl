__kernel void adder(__global const uchar* d_frame_in1, 
					__global const uchar* d_frame_in2, 
					__global const int* size1,
					__global const int* size2,
					__global uchar* d_frame_out)
{
	int idx = get_global_id(0);

	int w1 = size1[1];
	int h1 = size1[0];
	int w2 = size2[1];
	int h2 = size2[0];
	
	float index = convert_float(idx);

	int x1=idx%w1;
	int y1=convert_int((index-convert_float(x1))/w1);
	int tmp = (y1*w1+x1)*3;

	int y2=y1-98;
	int x2=x1-98;
	int tmp2 = (y2*w2+x2)*3;

	int tmpOut = tmp;

	if (x2>=0 && y2>=0 && x2<w2 && y2<h2){
		d_frame_out[tmpOut] = d_frame_in2[tmp2];
		d_frame_out[tmpOut+1] = d_frame_in2[tmp2+1];
		d_frame_out[tmpOut+2] = d_frame_in2[tmp2+2];
	} else {
		d_frame_out[tmpOut] = d_frame_in1[tmp];
		d_frame_out[tmpOut+1] = d_frame_in1[tmp+1];
		d_frame_out[tmpOut+2] = d_frame_in1[tmp+2];
	}
}