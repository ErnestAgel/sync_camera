#include <sys/time.h>
#include "cuda_runtime.h"
#include "format_conv.h"

#define RGB2Y(R, G, B)  ((77 * (R) + 150 * (G) + 29 * (B) +128) >> 8 )  
#define RGB2U(R, G, B)  ((-43 * (R) -84 * (G) + 127 * (B) >> 8 )+ 128)  
#define RGB2V(R, G, B)  (((127 * (R) - 106 * (G) - 21 * (B) + 128) >> 8)+ 128)
#define CLIPVALUE(x, minValue, maxValue) ((x) < (minValue) ? (minValue) : ((x) > (maxValue) ? (maxValue) : (x)))

static __device__ const unsigned char uchar_clipping_table[] = {
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, // -128 - -121
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, // -120 - -113
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, // -112 - -105
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, // -104 -  -97
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -96 -  -89
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -88 -  -81
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -80 -  -73
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -72 -  -65
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -64 -  -57
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -56 -  -49
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -48 -  -41
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -40 -  -33
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -32 -  -25
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -24 -  -17
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //  -16 -   -9
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, //   -8 -   -1
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
	31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
	60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
	89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
	114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
	137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
	160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
	183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
	206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
	229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
	252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, // 256-263
	255, 255, 255, 255, 255, 255, 255, 255, // 264-271
	255, 255, 255, 255, 255, 255, 255, 255, // 272-279
	255, 255, 255, 255, 255, 255, 255, 255, // 280-287
	255, 255, 255, 255, 255, 255, 255, 255, // 288-295
	255, 255, 255, 255, 255, 255, 255, 255, // 296-303
	255, 255, 255, 255, 255, 255, 255, 255, // 304-311
	255, 255, 255, 255, 255, 255, 255, 255, // 312-319
	255, 255, 255, 255, 255, 255, 255, 255, // 320-327
	255, 255, 255, 255, 255, 255, 255, 255, // 328-335
	255, 255, 255, 255, 255, 255, 255, 255, // 336-343
	255, 255, 255, 255, 255, 255, 255, 255, // 344-351
	255, 255, 255, 255, 255, 255, 255, 255, // 352-359
	255, 255, 255, 255, 255, 255, 255, 255, // 360-367
	255, 255, 255, 255, 255, 255, 255, 255, // 368-375
	255, 255, 255, 255, 255, 255, 255, 255, // 376-383
};

/** Clip a value to the range 0<val<255. For speed this is done using an
 * array, so can only cope with numbers in the range -128<val<383.
 */
static __device__ unsigned char clipvalue(int val)
{
	// Old method (if)
	/* val = val < 0 ? 0 : val; */
	/* return val > 255 ? 255 : val; */
	
	// New method (array)
	const int clipping_table_offset = 128;
	return uchar_clipping_table[val + clipping_table_offset];
}

static __device__ void yuv2rgb(const unsigned char y, const unsigned char u, const unsigned char v, unsigned char* r,
                    unsigned char* g, unsigned char* b)
{
	const int y2 = (int)y;
	const int u2 = (int)u - 128;
	const int v2 = (int)v - 128;
	//std::cerr << "YUV=("<<y2<<","<<u2<<","<<v2<<")"<<std::endl;
	
	// This is the normal YUV conversion, but
	// appears to be incorrect for the firewire cameras
	/* int r2 = y2 + ( (v2*91947) >> 16); */
	/* int g2 = y2 - ( ((u2*22544) + (v2*46793)) >> 16 ); */
	/* int b2 = y2 + ( (u2*115999) >> 16); */

	// This is an adjusted version (UV spread out a bit)
	int r2 = y2 + ((v2 * 37221) >> 15);
	int g2 = y2 - (((u2 * 12975) + (v2 * 18949)) >> 15);
	int b2 = y2 + ((u2 * 66883) >> 15);
	//std::cerr << "   RGB=("<<r2<<","<<g2<<","<<b2<<")"<<std::endl;
	
	// Cap the values.
	*r = clipvalue(r2);
	*g = clipvalue(g2);
	*b = clipvalue(b2);
}

__global__ void yuyv2rgb(unsigned char *yuyv, unsigned char *rgb)
{
	unsigned char y0, y1, u, v;
	unsigned char r0, g0, b0;
	unsigned char r1, g1, b1;
	
	int nIn = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
	y0 = (unsigned char)yuyv[nIn];
	u  = (unsigned char)yuyv[nIn + 1];
	y1 = (unsigned char)yuyv[nIn + 2];
	v  = (unsigned char)yuyv[nIn + 3];
	
	yuv2rgb(y0, u, v, &r0, &g0, &b0);
	yuv2rgb(y1, u, v, &r1, &g1, &b1);
	
	int nOut = blockIdx.x * blockDim.x * 6 + threadIdx.x * 6;
	rgb[nOut] = r0;
	rgb[nOut + 1] = g0;
	rgb[nOut + 2] = b0;
	rgb[nOut + 3] = r1;
	rgb[nOut + 4] = g1;
	rgb[nOut + 5] = b1;
}

__global__ void yuv2rgb(unsigned char *yuv, unsigned char *rgb, int widht, int height,int num)
{

#if 0
        int slice = blockIdx.x * blockDim.x *num + threadIdx.x*num;
        unsigned char y, u, v;
	unsigned char r, g, b;
        int h;
        int w;
         int ypos;
         int upos;
        int vpos;
         int out;

        for(int i = 0; i < num;i++)
        {
                  h = slice/widht;
                  w = slice%widht;
                   ypos  = slice;
                   upos =  widht*height + (h/2)*widht/2 +(w>>1);
                   vpos =   widht*height + widht*height/4 + (h/2)*widht/2 + (w>>1) ;

                     y = yuv[ypos  ];
                    u = yuv[upos];
                    v =  yuv[vpos ];
                    yuv2rgb(y, u, v, &r, &g, &b);

                   out = blockIdx.x * blockDim.x * 3*NUM + threadIdx.x * 3*NUM;
                   rgb[out+i*3] = r;
                   rgb[out+i*3+1] = g;
                   rgb[out+i*3+2] = b;
        }
#else
	int slice = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned char y, u, v;
	unsigned char r, g, b;

        int h = slice/widht;
        int w = slice%widht;

         int ypos  = slice;
         int upos =  widht*height + (h/2)*widht/2  + (w>>1);
        int vpos =   widht*height + widht*height/4 + (h/2)*widht/2 +  (w>>1);

          y = yuv[ypos];
         u = yuv[upos];
         v = yuv[vpos ];
         yuv2rgb(y, u, v, &r, &g, &b);


         int out = blockIdx.x * blockDim.x * 3 + threadIdx.x * 3;
        rgb[out++] = r;
        rgb[out++] = g;
        rgb[out++] = b;
#endif
}

__global__ void yuv2abgr(unsigned char *yuv, unsigned char *abgr, int widht, int height,int num)
{
	int slice = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char y, u, v;
	unsigned char r, g, b;

    int h = slice/widht;
    int w = slice%widht;

    int ypos  = slice;
    int upos =  widht*height + (h/2)*widht/2  + (w>>1);
    int vpos =   widht*height + widht*height/4 + (h/2)*widht/2 +  (w>>1);

    y = yuv[ypos];
    u = yuv[upos];
    v = yuv[vpos ];
    yuv2rgb(y, u, v, &r, &g, &b);


    int out = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    abgr[out++] = 0;
    abgr[out++] = b;
    abgr[out++] = g;
    abgr[out++] = r;
}


static __device__ void process_cuda(const unsigned char *bayer, unsigned char * ydst, int width, int height, int start_with_green, int blue_line, int ix, int iy)
{
    ydst +=  width;
//    bayer += width;
    //printf("bayer address(0x%p), height(%d)", bayer, height);
    if (iy > 0 && iy < height -1 ){
        if(iy % 2 != 0){
            start_with_green = start_with_green;
            blue_line = blue_line;
        }
        else{
            start_with_green = !start_with_green;
            blue_line = !blue_line;
        }
        int t0,t1;
        bayer = bayer + width * (iy - 1);
        const unsigned char *bayer_end = bayer + (width - 2);
      
        if (start_with_green) {
            if (ix == 0){
                t0 = bayer[1] + bayer[width * 2 + 1];
                /* Write first pixel */
                t1 = bayer[0] + bayer[width * 2] + bayer[width + 1];
                if (blue_line){
                    ydst = ydst + (iy - 1) * width ;
                    *ydst = (8453 * bayer[width] + 5516 * t1 + 1661 * t0 + 524288) >> 15;
                }
                else{
                    ydst = ydst + (iy - 1) * width ;
                    *ydst = (4226 * t0 + 5516 * t1 + 3223 * bayer[width] + 524288) >> 15;
                }

                /* Write second pixel */
                t1 = bayer[width] + bayer[width + 2];
                if (blue_line){
                    ydst = ydst + 1; 
                    *ydst = (4226 * t1 + 16594 * bayer[width + 1] + 1611 * t0 + 524288) >> 15;
                }
                else{
                    ydst = ydst + 1; 
                    *ydst = (4226 * t0 + 16594 * bayer[width + 1] + 1611 * t1 + 524288) >> 15;
                }
            }
        }

        else {
            if(ix == 0){
                t0 = bayer[0] + bayer[width * 2];
                if (blue_line) {
                    ydst = ydst + (iy - 1) * width ;
                    *ydst = (8453 * bayer[width + 1] + 16594 * bayer[width] + 1661 * t0 + 524288) >> 15;
                } 
                else {
                    ydst = ydst  + (iy - 1) * width;
                    *ydst = (4226 * t0 + 16594 * bayer[width] + 3223 * bayer[width + 1] + 524288) >> 15;
                }
            }   
        }

        if(start_with_green){
            bayer = bayer  + 1;
            if (blue_line) {
                if(ix > 0 && ix < width - 3 && ix % 2 != 0){
                    ydst = ydst + (iy - 1) * width ;
                    ydst =ydst + 2;
                    bayer =bayer + 2 * ((ix - 1)/ 2 );
                    t0 = bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2];
                    t1 = bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1];
                    ydst = ydst + (ix - 1) ;
                    *ydst = (8453 * bayer[width + 1] + 4148 * t1 + 806 * t0 + 524288) >> 15;
                    ydst =ydst +1;
                    t0 = bayer[2] + bayer[width * 2 + 2];
                    t1 = bayer[width + 1] + bayer[width + 3];
                    *ydst = (4226 * t1 + 16594 * bayer[width + 2] + 1611 * t0 + 524288) >> 15;
                }
            } 
            else{
                if(ix > 0 && ix < width -3 && ix % 2!= 0){
                    ydst = ydst + (iy - 1) * width ;
                    ydst =ydst + 2;
                    bayer =bayer + 2 * ((ix - 1)/ 2 );
                    t0 = bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2];
                    t1 = bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1];
                    ydst = ydst + (ix - 1) ;
                    *ydst = (2113 * t0 + 4148 * t1 + 3223 * bayer[width + 1] + 524288) >> 15;
                    ydst =ydst +1;
                    t0 = bayer[2] + bayer[width * 2 + 2];
                    t1 = bayer[width + 1] + bayer[width + 3];
                    *ydst = (4226 * t0 + 16594 * bayer[width + 2] + 1611 * t1 + 524288) >> 15;
                }
            }

            // if (bayer < bayer_end) {
            //     if(ix == width - 5){
            //          /* Write second to last pixel */
            //         t0 = bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2];
            //         t1 = bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1];
            //         if (blue_line){
            //             ydst = ydst + 1;
            //             *ydst = (8453 * bayer[width + 1] + 16594 * bayer[width + 2] + 1661 * t0 + 524288) >> 15;
            //         }
            //         else{
            //             ydst = ydst + 1;
            //             *ydst = (2113 * t0 + 4148 * t1 + 3223 * bayer[width + 1] + 524288) >> 15;
            //         }
            //         /* write last pixel */
            //         t0 = bayer[2] + bayer[width * 2 + 2];
            //         if (blue_line) {
            //             ydst = ydst + 1;
            //             *ydst = (8453 * bayer[width + 1] + 16594 * bayer[width + 2] + 1661 * t0 + 524288) >> 15;
            //         } else {
            //             ydst = ydst + 1;
            //             *ydst = (4226 * t0 + 16594 * bayer[width + 2] + 3223 * bayer[width + 1] + 524288) >> 15;
            //         }
            //     }
            // }
            // else{
            //     if(ix == width -5){
            //     /* write last pixel */
                
            //         t0 = bayer[0] + bayer[width * 2];
            //         t1 = bayer[1] + bayer[width * 2 + 1] + bayer[width];
            //         if (blue_line){
            //             ydst = ydst + 1;
            //             *ydst = (8453 * bayer[width + 1] + 5516 * t1 + 1661 * t0 + 524288) >> 15;
            //         }
            //         else{
            //             ydst = ydst + 1;
            //             *ydst = (4226 * t0 + 5516 * t1 + 3223 * bayer[width + 1] + 524288) >> 15;
            //         }     
            //     }   
            // }
        }
        else{
            if (blue_line) {
                if(ix >= 0 && ix < width -3 && ix % 2 != 0){
                    ydst = ydst + (iy - 1) * width ;
                    ydst =ydst + 1;
                    bayer =bayer  + 2 * ((ix-1 )/ 2 );
                    t0 = bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2];
                    t1 = bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1];
                    ydst = ydst + (ix -1) ;
                    *ydst = (8453 * bayer[width + 1] + 4148 * t1 + 806 * t0 + 524288) >> 15;
                    ydst =ydst +1;
                    t0 = bayer[2] + bayer[width * 2 + 2];
                    t1 = bayer[width + 1] + bayer[width + 3];
                    *ydst = (4226 * t1 + 16594 * bayer[width + 2] + 1611 * t0 + 524288) >> 15;
                }
            } 
            else{
                if(ix >= 0 && ix < width -3 && ix % 2 != 0){
                     ydst = ydst + (iy - 1) * width ;
                    ydst =ydst + 1;
                    bayer =bayer  + 2 * ((ix-1 )/ 2 );
                    t0 = bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2];
                    t1 = bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1];
                    ydst = ydst + (ix -1) ;
                    *ydst = (2113 * t0 + 4148 * t1 + 3223 * bayer[width + 1] + 524288) >> 15;
                    ydst =ydst +1;
                    t0 = bayer[2] + bayer[width * 2 + 2];
                    t1 = bayer[width + 1] + bayer[width + 3];
                    *ydst = (4226 * t0 + 16594 * bayer[width + 2] + 1611 * t1 + 524288) >> 15;
                }
            }

            if (bayer < bayer_end) {

                if(ix == width - 5){
                   // printf("****%d****", ix);
                    t0 = bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2];
                    t1 = bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1];
                    if (blue_line){
                        ydst = ydst + 1;
                        *ydst = (8453 * bayer[width + 1] + 16594 * bayer[width + 2] + 1661 * t0 + 524288) >> 15;
                    }
                    else{
                        ydst = ydst + 1;
                        *ydst = (2113 * t0 + 4148 * t1 + 3223 * bayer[width + 1] + 524288) >> 15;
                    }

                    // t0 = bayer[2] + bayer[width * 2 + 2];
                    // if (blue_line) {
                    //     ydst = ydst + 1;
                    //     *ydst = (8453 * bayer[width + 1] + 16594 * bayer[width + 2] + 1661 * t0 + 524288) >> 15;
                    // } else {
                    //     ydst = ydst + 1;
                    //     *ydst = (4226 * t0 + 16594 * bayer[width + 2] + 3223 * bayer[width + 1] + 524288) >> 15;
                    // }
                }
            }
            else{
                if(ix == width - 5){
//                    bayer = bayer + width - 3;
                    /* write last pixel */
                    t0 = bayer[0] + bayer[width * 2];
                    t1 = bayer[1] + bayer[width * 2 + 1] + bayer[width];
                    if (blue_line){
                        ydst = ydst + 1;
                        *ydst = (8453 * bayer[width + 1] + 5516 * t1 + 1661 * t0 + 524288) >> 15;
                    }
                    else{
                        ydst = ydst + 1;
                        *ydst++ = (4226 * t0 + 5516 * t1 + 3223 * bayer[width + 1] + 524288) >> 15;
                    }  
                }      
            }
        }
    }
}


static void v4lconvert_border_bayer_line_to_y( const unsigned char *bayer, const unsigned char *adjacent_bayer, unsigned char *y, int width, int start_with_green, int blue_line)
{
        int t0, t1;
    
        if (start_with_green) {
            /* First pixel */
            if (blue_line) {
                *y++ = (8453 * adjacent_bayer[0] + 16594 * bayer[0] + 3223 * bayer[1] + 524288) >> 15;
            } 
            else {
                *y++ = (8453 * bayer[1] + 16594 * bayer[0] + 3223 * adjacent_bayer[0] + 524288) >> 15;
            }
            /* Second pixel */
            t0 = bayer[0] + bayer[2] + adjacent_bayer[1];
            t1 = adjacent_bayer[0] + adjacent_bayer[2];
            if (blue_line)
                *y++ = (4226 * t1 + 5531 * t0 + 3223 * bayer[1] + 524288) >> 15;
            else
                *y++ = (8453 * bayer[1] + 5531 * t0 + 1611 * t1 + 524288) >> 15;
            bayer++;
            adjacent_bayer++;
            width -= 2;
        } 
        else {
            /* First pixel */
            t0 = bayer[1] + adjacent_bayer[0];
            if (blue_line) {
                *y++ = (8453 * adjacent_bayer[1] + 8297 * t0 + 3223 * bayer[0] + 524288) >> 15;
            } 
            else {
                *y++ = (8453 * bayer[0] + 8297 * t0 + 3223 * adjacent_bayer[1] + 524288) >> 15;
            }
            width--;
        }
        if (blue_line) {
            for ( ; width > 2; width -= 2) {
                t0 = bayer[0] + bayer[2];
                *y++ = (8453 * adjacent_bayer[1] + 16594 * bayer[1] + 1611 * t0 + 524288) >> 15;
                bayer++;
                adjacent_bayer++;
    
                t0 = bayer[0] + bayer[2] + adjacent_bayer[1];
                t1 = adjacent_bayer[0] + adjacent_bayer[2];
                *y++ = (4226 * t1 + 5531 * t0 + 3223 * bayer[1] + 524288) >> 15;
                bayer++;
                adjacent_bayer++;
            }
        } 
        else {
            for ( ; width > 2; width -= 2) {
                t0 = bayer[0] + bayer[2];
                *y++ = (4226 * t0 + 16594 * bayer[1] + 3223 * adjacent_bayer[1] + 524288) >> 15;
                bayer++;
                adjacent_bayer++;
    
                t0 = bayer[0] + bayer[2] + adjacent_bayer[1];
                t1 = adjacent_bayer[0] + adjacent_bayer[2];
                *y++ = (8453 * bayer[1] + 5531 * t0 + 1611 * t1 + 524288) >> 15;
                bayer++;
                adjacent_bayer++;
            }
        }
    
        if (width == 2) {
            /* Second to last pixel */
            t0 = bayer[0] + bayer[2];
            if (blue_line) {
                *y++ = (8453 * adjacent_bayer[1] + 16594 * bayer[1] + 1611 * t0 + 524288) >> 15;
            } 
            else {
                *y++ = (4226 * t0 + 16594 * bayer[1] + 3223 * adjacent_bayer[1] + 524288) >> 15;
            }
            /* Last pixel */
            t0 = bayer[1] + adjacent_bayer[2];
            if (blue_line) {
                *y++ = (8453 * adjacent_bayer[1] + 8297 * t0 + 3223 * bayer[2] + 524288) >> 15;
            } 
            else {
                *y++ = (8453 * bayer[2] + 8297 * t0 + 3223 * adjacent_bayer[1] + 524288) >> 15;
            }
        } 
        else {
            /* Last pixel */
            if (blue_line) {
                *y++ = (8453 * adjacent_bayer[1] + 16594 * bayer[1] + 3223 * bayer[0] + 524288) >> 15;
            } 
            else {
                *y++ = (8453 * bayer[0] + 16594 * bayer[1] + 3223 * adjacent_bayer[1] + 524288) >> 15;
            }
        }
}


__global__  void bayer2yuv(const unsigned char *bayer, unsigned char *yuv, int width, int height)
{
    int blue_line =0, start_with_green = 0;
    unsigned char *ydst = yuv;
    unsigned char *udst, *vdst;

    if (0) {
        vdst = yuv + width * height;
        udst = vdst + width * height / 4;
    } 
    else {
        udst = yuv + width * height;
        vdst = udst + width * height / 4;
    }

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(ix % 2 ==0 && iy % 2 == 0){
        int b, g, r;
        bayer = bayer + 2 * width * (iy / 2);
        r = bayer[ix];
        g = bayer[ix + 1];
        g += bayer[ix + width];
        b = bayer[ix + width +1];
        udst = udst + (iy / 2) * (width / 2) + ix / 2;
        vdst = vdst + (iy / 2) * (width / 2) + ix / 2;
        *udst = (-4878 * r - 4789 * g + 14456 * b + 4210688) >> 15;
        *vdst = (14456 * r - 6052 * g -  2351 * b + 4210688) >> 15;
         bayer = bayer - 2 * width * (iy / 2);
    }
    process_cuda(bayer, ydst, width, height, blue_line, start_with_green, ix, iy);
}

static void v4lconvert_border_bayer_line_to_bgr24(
	const unsigned char *bayer, const unsigned char *adjacent_bayer,
	unsigned char *bgr, int width, int start_with_green, int blue_line)
{
	int t0, t1;

	if (start_with_green)
	{
		/* First pixel */
		if (blue_line)
		{
			*bgr++ = bayer[1];
			*bgr++ = bayer[0];
			*bgr++ = adjacent_bayer[0];
		}
		else
		{
			*bgr++ = adjacent_bayer[0];
			*bgr++ = bayer[0];
			*bgr++ = bayer[1];
		}
		/* Second pixel */
		t0 = (bayer[0] + bayer[2] + adjacent_bayer[1] + 1) / 3;
		t1 = (adjacent_bayer[0] + adjacent_bayer[2] + 1) >> 1;
		if (blue_line)
		{
			*bgr++ = bayer[1];
			*bgr++ = t0;
			*bgr++ = t1;
		}
		else
		{
			*bgr++ = t1;
			*bgr++ = t0;
			*bgr++ = bayer[1];
		}
		bayer++;
		adjacent_bayer++;
		width -= 2;
	}
	else
	{
		/* First pixel */
		t0 = (bayer[1] + adjacent_bayer[0] + 1) >> 1;
		if (blue_line)
		{
			*bgr++ = bayer[0];
			*bgr++ = t0;
			*bgr++ = adjacent_bayer[1];
		}
		else
		{
			*bgr++ = adjacent_bayer[1];
			*bgr++ = t0;
			*bgr++ = bayer[0];
		}
		width--;
	}

	if (blue_line)
	{
		for (; width > 2; width -= 2)
		{
			t0 = (bayer[0] + bayer[2] + 1) >> 1;
			*bgr++ = t0;
			*bgr++ = bayer[1];
			*bgr++ = adjacent_bayer[1];
			bayer++;
			adjacent_bayer++;

			t0 = (bayer[0] + bayer[2] + adjacent_bayer[1] + 1) / 3;
			t1 = (adjacent_bayer[0] + adjacent_bayer[2] + 1) >> 1;
			*bgr++ = bayer[1];
			*bgr++ = t0;
			*bgr++ = t1;
			bayer++;
			adjacent_bayer++;
		}
	}
	else
	{
		for (; width > 2; width -= 2)
		{
			t0 = (bayer[0] + bayer[2] + 1) >> 1;
			*bgr++ = adjacent_bayer[1];
			*bgr++ = bayer[1];
			*bgr++ = t0;
			bayer++;
			adjacent_bayer++;

			t0 = (bayer[0] + bayer[2] + adjacent_bayer[1] + 1) / 3;
			t1 = (adjacent_bayer[0] + adjacent_bayer[2] + 1) >> 1;
			*bgr++ = t1;
			*bgr++ = t0;
			*bgr++ = bayer[1];
			bayer++;
			adjacent_bayer++;
		}
	}

	if (width == 2)
	{
		/* Second to last pixel */
		t0 = (bayer[0] + bayer[2] + 1) >> 1;
		if (blue_line)
		{
			*bgr++ = t0;
			*bgr++ = bayer[1];
			*bgr++ = adjacent_bayer[1];
		}
		else
		{
			*bgr++ = adjacent_bayer[1];
			*bgr++ = bayer[1];
			*bgr++ = t0;
		}
		/* Last pixel */
		t0 = (bayer[1] + adjacent_bayer[2] + 1) >> 1;
		if (blue_line)
		{
			*bgr++ = bayer[2];
			*bgr++ = t0;
			*bgr++ = adjacent_bayer[1];
		}
		else
		{
			*bgr++ = adjacent_bayer[1];
			*bgr++ = t0;
			*bgr++ = bayer[2];
		}
	}
	else
	{
		/* Last pixel */
		if (blue_line)
		{
			*bgr++ = bayer[0];
			*bgr++ = bayer[1];
			*bgr++ = adjacent_bayer[1];
		}
		else
		{
			*bgr++ = adjacent_bayer[1];
			*bgr++ = bayer[1];
			*bgr++ = bayer[0];
		}
	}
}

__global__ void bayer2rgb(const unsigned char *bayer, unsigned char *bgr, int width, int height)
{
    int start_with_green =0;
    int blue_line = 1;

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;


    bgr += width * 3;


    if (iy > 0 && iy < height -1 ){
        if(iy % 2 != 0){
            start_with_green = start_with_green;
            blue_line = blue_line;
        }
        else{
            start_with_green = !start_with_green;
            blue_line = !blue_line;
        }
        int t0,t1;
        bayer = bayer + width * (iy - 1);
        const unsigned char *bayer_end = bayer + (width - 2);

        if (start_with_green) {
            if (ix == 0){
                t0 = (bayer[1] + bayer[width * 2 + 1] + 1) >> 1;
                /* Write first pixel */
                t1 = (bayer[0] + bayer[width * 2] + bayer[width + 1] + 1) / 3;
                if(blue_line){
                    bgr = bgr + (iy - 1) * 3 * width ;
                    *bgr = t0;
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = bayer[width];
                }
                else{
                    bgr = bgr + (iy - 1) * 3 * width ;
                    *bgr = bayer[width];
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = t0;
                }

                /* Write second pixel */
                t1 = (bayer[width] + bayer[width + 2] + 1) >> 1;
                if (blue_line) {
                    bgr++;
                    *bgr = t0;
                    bgr++;
                    *bgr = bayer[width + 1];
                    bgr++;
                    *bgr = t1;
                } 
                else {
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = bayer[width + 1];
                    bgr++;
                    *bgr = t0;
                }
            }
        }
        else{
            if (ix == 0){
                    /* Write first pixel */
                    t0 = (bayer[0] + bayer[width * 2] + 1) >> 1;
                    if (blue_line) {
                        bgr = bgr + (iy - 1) * 3 * width ;
                        *bgr = t0;
                        bgr++;
                        *bgr = bayer[width];
                        bgr++;
                        *bgr = bayer[width + 1];
                    } 
                    else {
                        bgr = bgr + (iy - 1) * 3 * width ;
                        *bgr = bayer[width + 1];
                        bgr++;
                        *bgr = bayer[width];
                        bgr++;
                        *bgr = t0;
                    }
            }
        }
        if(start_with_green){
            bayer = bayer  + 1;
            if (blue_line) {
                if(ix > 0 && ix < width -3 && ix % 2 != 0){
                    bayer =bayer + 2 * ((ix - 1)/ 2 );
                    t0 = (bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2] + 2) >> 2;
                    t1 = (bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1] + 2) >> 2;
                    bgr = bgr + (iy - 1) * 3 * width ;
                    bgr = bgr + 2 * 3;
                    bgr = bgr + 3 * (ix - 1);
                    *bgr = t0;
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = bayer[width + 1];
                    
                    t0 = (bayer[2] + bayer[width * 2 + 2] + 1) >> 1;
                    t1 = (bayer[width + 1] + bayer[width + 3] + 1) >> 1;
                    bgr++;
                    *bgr = t0;
                    bgr++;
                    *bgr = bayer[width + 2];
                    bgr++;
                    *bgr = t1;
                }
            }
            else{
                if(ix > 0 && ix < width -3 && ix % 2 != 0){
                    bayer =bayer + 2 * ((ix - 1)/ 2 );
                    t0 = (bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2] + 2) >> 2;
                    t1 = (bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1] + 2) >> 2;
                    bgr = bgr + (iy - 1) * 3 * width ;
                    bgr = bgr + 2 * 3;
                    bgr = bgr + 3 * (ix - 1);
                    *bgr = bayer[width + 1];
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = t0;
                    
                    t0 = (bayer[2] + bayer[width * 2 + 2] + 1) >> 1;
                    t1 = (bayer[width + 1] + bayer[width + 3] + 1) >> 1;
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = bayer[width + 2];
                    bgr++;
                    *bgr = t0;
                }
            }

            if (bayer < bayer_end) {
                if(ix == width - 4){
                     /* Write second to last pixel */
                    t0 = (bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2] + 2) >> 2;
                    t1 = (bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1] + 2) >> 2;
                    if (blue_line){
                        bgr++;
                        *bgr = t0;
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = bayer[width + 1];
                    }
                    else{
                        bgr++;
                        *bgr = bayer[width + 1];
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = t0;
                    }
                    /* write last pixel */
                    t0 = bayer[2] + bayer[width * 2 + 2];
                    if (blue_line) {
                        bgr++;
                        *bgr = t0;
                        bgr++;
                        *bgr = bayer[width + 2];
                        bgr++;
                        *bgr = bayer[width + 1];
                    } else {
                        bgr++;
                        *bgr = bayer[width + 2];
                        bgr++;
                        *bgr = bayer[width + 1];
                        bgr++;
                        *bgr = t0;
                    }
                }
            }
            else{
                if(ix == width - 4){
                /* write last pixel */
                    t0 = (bayer[0] + bayer[width * 2] + 1) >> 1;
                    t1 = (bayer[1] + bayer[width * 2 + 1] + bayer[width] + 1) / 3;
                    if (blue_line){
                        bgr++;
                        *bgr = t0;
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = bayer[width + 1];
                    }
                    else{
                        bgr++;
                        *bgr = bayer[width + 1];
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = t0;
                    }     
                }   
            }
        }
        else{
            if (blue_line) {
                if(ix > 0 && ix < width - 2 && ix % 2 != 0){
                    bayer =bayer + 2 * ((ix - 1)/ 2 );
                    t0 = (bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2] + 2) >> 2;
                    t1 = (bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1] + 2) >> 2;
                    bgr = bgr + (iy - 1) * 3 * width ;
                    bgr = bgr + 1 * 3;
                    bgr = bgr + 3 * (ix - 1);
                    *bgr = t0;
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = bayer[width + 1];
                    
                    t0 = (bayer[2] + bayer[width * 2 + 2] + 1) >> 1;
                    t1 = (bayer[width + 1] + bayer[width + 3] + 1) >> 1;
                    bgr++;
                    *bgr = t0;
                    bgr++;
                    *bgr = bayer[width + 2];
                    bgr++;
                    *bgr = t1;
                }
            }
            else{
                if(ix > 0 && ix < width - 2 && ix % 2 != 0){
                    bayer =bayer + 2 * ((ix - 1)/ 2 );
                    t0 = (bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2] + 2) >> 2;
                    t1 = (bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1] + 2) >> 2;
                    bgr = bgr + (iy - 1) * 3 * width ;
                    bgr = bgr + 1 * 3;
                    bgr = bgr + 3 * (ix - 1);
                    *bgr = bayer[width + 1];
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = t0;
                    
                    t0 = (bayer[2] + bayer[width * 2 + 2] + 1) >> 1;
                    t1 = (bayer[width + 1] + bayer[width + 3] + 1) >> 1;
                    bgr++;
                    *bgr = t1;
                    bgr++;
                    *bgr = bayer[width + 2];
                    bgr++;
                    *bgr = t0;
                }
            }

            if (bayer < bayer_end) {
                if(ix == width - 4){
                     /* Write second to last pixel */
                    t0 = (bayer[0] + bayer[2] + bayer[width * 2] + bayer[width * 2 + 2] + 2) >> 2;
                    t1 = (bayer[1] + bayer[width] + bayer[width + 2] + bayer[width * 2 + 1] + 2) >> 2;
                    if (blue_line){
                        bgr++;
                        *bgr = t0;
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = bayer[width + 1];
                    }
                    else{
                        bgr++;
                        *bgr = bayer[width + 1];
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = t0;
                    }
                    /* write last pixel */
                    t0 = bayer[2] + bayer[width * 2 + 2];
                    if (blue_line) {
                        bgr++;
                        *bgr = t0;
                        bgr++;
                        *bgr = bayer[width + 2];
                        bgr++;
                        *bgr = bayer[width + 1];
                    } else {
                        bgr++;
                        *bgr = bayer[width + 2];
                        bgr++;
                        *bgr = bayer[width + 1];
                        bgr++;
                        *bgr = t0;
                    }
                }
            }
            else{
                if(ix == width - 4){
                /* write last pixel */
                    t0 = (bayer[0] + bayer[width * 2] + 1) >> 1;
                    t1 = (bayer[1] + bayer[width * 2 + 1] + bayer[width] + 1) / 3;
                    if (blue_line){
                        bgr++;
                        *bgr = t0;
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = bayer[width + 1];
                    }
                    else{
                        bgr++;
                        *bgr = bayer[width + 1];
                        bgr++;
                        *bgr = t1;
                        bgr++;
                        *bgr = t0;
                    }     
                }   
            }
        }
    }
}

__global__ void rgb2yuv(unsigned char* rgb, unsigned char* yuv, int width, int height)
{
    int slice = blockIdx.x * blockDim.x * 3 + threadIdx.x * 3;
    unsigned char r, g, b, y, u, v;
    r = rgb[slice++];
    g = rgb[slice++];
    b = rgb[slice++];

    y = (unsigned char)(CLIPVALUE(RGB2Y(r, g, b), 0, 255));
    u = (unsigned char)(CLIPVALUE(RGB2U(r, g, b), 0, 255));
    v = (unsigned char)(CLIPVALUE(RGB2V(r, g, b), 0, 255));

    int nOut = blockIdx.x * blockDim.x + threadIdx.x;
    
    int h = nOut / width;
    int w = nOut % width;

    int ypos = nOut;
    int upos = width * height + (h / 2) * width / 2 + (w >> 1);
    int vpos = width * height + width * height / 4 + (h / 2) * width / 2 + (w >> 1);

    yuv[ypos] = y;
    yuv[upos] = u;
    yuv[vpos] = v;

}

__global__ void abgr2yuv(unsigned char* abgr, unsigned char* yuv, int width, int height)
{
    int slice = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    unsigned char r, g, b, y, u, v;
    slice++;
    b = abgr[slice++];
    g = abgr[slice++];
    r = abgr[slice++];

    y = (unsigned char)(CLIPVALUE(RGB2Y(r, g, b), 0, 255));
    u = (unsigned char)(CLIPVALUE(RGB2U(r, g, b), 0, 255));
    v = (unsigned char)(CLIPVALUE(RGB2V(r, g, b), 0, 255));

    int nOut = blockIdx.x * blockDim.x + threadIdx.x;
    
    int h = nOut / width;
    int w = nOut % width;

    int ypos = nOut;
    int upos = width * height + (h / 2) * width / 2 + (w >> 1);
    int vpos = width * height + width * height / 4 + (h / 2) * width / 2 + (w >> 1);

    yuv[ypos] = y;
    yuv[upos] = u;
    yuv[vpos] = v;
}

LIBFORMATCONV_API   void CUDA_rgb_to_yu12(unsigned char* rgb, unsigned char* yuv, int width, int height)
{
    int block_size = 256;
    int num_blocks = width * height / block_size;
    rgb2yuv<<<num_blocks, block_size>>>(rgb, yuv, width, height);
    cudaDeviceSynchronize();
}

LIBFORMATCONV_API   void CUDA_abgr_to_yu12(unsigned char* abgr, unsigned char* yuv, int width, int height)
{
    int block_size = 256;
    int num_blocks = width * height / block_size;
    abgr2yuv<<<num_blocks, block_size>>>(abgr, yuv, width, height);
    cudaDeviceSynchronize();
}


LIBFORMATCONV_API  void CALLBACK CUDA_yu12_to_rgb(unsigned char *yuv, unsigned char *rgb, int width, int height)
{
	int block_size = 512;
	int num_blocks = width*height/block_size;
	yuv2rgb<<<num_blocks,block_size>>>(yuv, rgb, width, height,1);
	cudaDeviceSynchronize();
}

LIBFORMATCONV_API   void CALLBACK CUDA_yu12_to_abgr(unsigned char *yuv, unsigned char *abgr, int width, int height)
{
    int block_size = 512;
	int num_blocks = width*height/block_size;
	yuv2abgr<<<num_blocks,block_size>>>(yuv, abgr, width, height,1);
	cudaDeviceSynchronize();
}

LIBFORMATCONV_API void CALLBACK CUDA_yuyv_to_rgb(unsigned char *yuyv, unsigned char *rgb, int width, int height)
{
	int block_size = 256;
	int num_blocks = (2*width*height)/(block_size*4);
	yuyv2rgb<<<num_blocks, block_size>>>(yuyv, rgb);
	cudaDeviceSynchronize();
}

LIBFORMATCONV_API   void CALLBACK CUDA_bayerrg8_to_rgb(unsigned char *bayer, unsigned char *rgb, int width, int height)
{
	cudaStreamAttachMemAsync(NULL, bayer, 0, cudaMemAttachGlobal);
//	cudaMemcpy(bayer_cuda, bayer, src_size, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(16, 16);
    	dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	bayer2rgb<<<blocksPerGrid, threadsPerBlock>>>(bayer, rgb, width, height);
    	cudaStreamAttachMemAsync(NULL, rgb, 0, cudaMemAttachHost);
    	cudaStreamSynchronize(NULL);
//	cudaDeviceSynchronize();
//	cudaMemcpy(rgb, rgb_cuda,  dest_size, cudaMemcpyDeviceToHost);
        v4lconvert_border_bayer_line_to_bgr24(bayer, bayer + width, rgb, width, 0, 0);
        v4lconvert_border_bayer_line_to_bgr24(bayer + width * (height - 1) , bayer + width * (height - 2), rgb + 3 * width * (height - 1), width, 1, 0);
}

LIBFORMATCONV_API   void CALLBACK CUDA_bayerrg8_to_yu12(unsigned char *bayer, unsigned char *yuv, int width, int height)
{
	cudaStreamAttachMemAsync(NULL, bayer, 0, cudaMemAttachGlobal);
//	cudaMemcpy(bayer_cuda, bayer, src_size, cudaMemcpyHostToDevice);
	dim3 threadsPerBlock(16, 16);
    	dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
	bayer2yuv<<<blocksPerGrid, threadsPerBlock>>>(bayer, yuv, width, height);
//	cudaDeviceSynchronize();
//	cudaMemcpy(yuv, yuv_cuda,  dest_size, cudaMemcpyDeviceToHost);
	cudaStreamAttachMemAsync(NULL, yuv, 0, cudaMemAttachHost);
    	cudaStreamSynchronize(NULL);
   //    v4lconvert_border_bayer_line_to_y(bayer, bayer + width, yuv, width, 0, 0);
   //     v4lconvert_border_bayer_line_to_y(bayer + width * (height - 1), bayer + width * (height - 2) , yuv +width * (height - 1), width, 1, 0);
}

#if 0
 void process_yuv_cuda(unsigned char *yuv, unsigned char *rgb, int width, int height)
{
        unsigned char *image_yuv_cuda_ = NULL;                //输入在GPU的内存
        unsigned char *image_rgb_cuda_ = NULL;             //输出在GPU的内存

    printf("%s\n",__func__);
         cudaMalloc((void **)&image_yuv_cuda_, 3*width*height/2);
          cudaMalloc((void **)&image_rgb_cuda_,3*width*height);


        cudaError_t ret = cudaMemcpy(image_yuv_cuda_, yuv, 3*width*height/2, cudaMemcpyHostToDevice);

        if (cudaSuccess != ret) {
                printf("cudaMemcpy fail %d\n", ret);
        }


        yuv2rgb_cuda(image_yuv_cuda_, image_rgb_cuda_, width,height);

        ret = cudaMemcpy(rgb, image_rgb_cuda_, 3*width*height, cudaMemcpyDeviceToHost);

        if (cudaSuccess != ret) {
                printf("cudaMemcpy fail %d\n", ret);
        }

	cudaFree(image_yuv_cuda_);
	cudaFree(image_rgb_cuda_);
}

#endif
/*
extern "C" void process_image_cuda(const void *src, int size)
{
        struct timeval ts;

        int yuv_size = size * sizeof(char);

        gettimeofday(&ts, NULL);
        printf("[%lu.%lu]\tbefore copy image_data(CPU to GPU)\n", ts.tv_sec, ts.tv_usec);
        cudaError_t ret = cudaMemcpy(image_yuyv_cuda_, src, yuv_size, cudaMemcpyHostToDevice);
        gettimeofday(&ts, NULL);
        printf("[%lu.%lu]\tcopy image_data(CPU to GPU) done\n", ts.tv_sec, ts.tv_usec);

        if (cudaSuccess != ret) {
                printf("cudaMemcpy fail %d\n", ret);
        }
        const int block_size = 256;
        const int num_blocks = yuv_size / (4*block_size);


        gettimeofday(&ts, NULL);
        printf("[%lu.%lu]\tbefore yuyv2rgb computation\n", ts.tv_sec, ts.tv_usec);
        yuyv2rgb_cuda(image_yuyv_cuda_, image_rgb_cuda_, num_blocks, block_size);
        gettimeofday(&ts, NULL);
        printf("[%lu.%lu]\tyuyv2rgb computation done\n", ts.tv_sec, ts.tv_usec);


        int rgb_size = size / 2 * 3 * sizeof(char);

        gettimeofday(&ts, NULL);
        printf("[%lu.%lu]\tbefore copy image_data(GPU to CPU)\n", ts.tv_sec, ts.tv_usec);
        ret = cudaMemcpy(show_buf, image_rgb_cuda_, rgb_size, cudaMemcpyDeviceToHost);
        gettimeofday(&ts, NULL);
        printf("[%lu.%lu]\tcopy image_data(GPU to CPU) done\n", ts.tv_sec, ts.tv_usec);
        printf("[%lu.%lu]\tcuda process image index = %d\n", ts.tv_sec, ts.tv_usec, ++index_pro);

        if (cudaSuccess != ret) {
                printf("cudaMemcpy fail %d\n", ret);
        }
}
*/
