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
	int tmpWidth = 50;
	
	float index = convert_float(idx);

	int x1=idx%w1;
	int y1=convert_int((index-convert_float(x1))/w1);
	int tmp = (y1*w1+x1)*3;

	int y2=y1-98;
	int x2=x1-98;
	int tmp2 = (y2*w2+x2)*3;
	int tmp3 = 0;

	int xout=h1-y1;
	int yout=x1;

	int tmpOut = (yout*h1+xout)*3;

	if (d_frame_in1[tmp]<40 && d_frame_in1[tmp+1]<40 && d_frame_in1[tmp+2]<40){
		if (x1<w1/2){
			if (y1<h1/2){
				if (x1<w1/2-tmpWidth){
					tmp3 = (y2*w2+(x2+tmpWidth))*3;
					d_frame_out[tmpOut] = d_frame_in2[tmp3];
					d_frame_out[tmpOut+1] = d_frame_in2[tmp3+1];
					d_frame_out[tmpOut+2] = d_frame_in2[tmp3+2];
				} else {
					d_frame_out[tmpOut] = 0;
					d_frame_out[tmpOut+1] = 0;
					d_frame_out[tmpOut+2] = 0;
				}
			} else {
				if (y1>h1/2+tmpWidth){
					tmp3 = ((y2-tmpWidth)*w2+x2)*3;
					d_frame_out[tmpOut] = d_frame_in2[tmp3];
					d_frame_out[tmpOut+1] = d_frame_in2[tmp3+1];
					d_frame_out[tmpOut+2] = d_frame_in2[tmp3+2];
				} else {
					d_frame_out[tmpOut] = 0;
					d_frame_out[tmpOut+1] = 0;
					d_frame_out[tmpOut+2] = 0;
				}
			}
		} else {
			if (y1<h1/2){
				if (y1<h1/2-tmpWidth){
					tmp3 = ((y2+tmpWidth)*w2+x2)*3;
					d_frame_out[tmpOut] = d_frame_in2[tmp3];
					d_frame_out[tmpOut+1] = d_frame_in2[tmp3+1];
					d_frame_out[tmpOut+2] = d_frame_in2[tmp3+2];
				} else {
					d_frame_out[tmpOut] = 0;
					d_frame_out[tmpOut+1] = 0;
					d_frame_out[tmpOut+2] = 0;
				}
			} else {
				if (x1>w1/2+tmpWidth){
					tmp3 = (y2*w2+(x2-tmpWidth))*3;
					d_frame_out[tmpOut] = d_frame_in2[tmp3];
					d_frame_out[tmpOut+1] = d_frame_in2[tmp3+1];
					d_frame_out[tmpOut+2] = d_frame_in2[tmp3+2];
				} else {
					d_frame_out[tmpOut] = 0;
					d_frame_out[tmpOut+1] = 0;
					d_frame_out[tmpOut+2] = 0;
				}
			}
		}
	} else {
		d_frame_out[tmpOut] = d_frame_in1[tmp];
		d_frame_out[tmpOut+1] = d_frame_in1[tmp+1];
		d_frame_out[tmpOut+2] = d_frame_in1[tmp+2];
	}
}